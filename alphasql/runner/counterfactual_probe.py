"""Bounded DPC-inspired SQL/Python checks for ambiguous fixed candidate pairs.

This is a constrained subset of DPC, not a reproduction of its full agent
pipeline. A tester proposes tiny synthetic tables that distinguish two SQLs;
a separate solver, without seeing either SQL, returns a restricted Python
expression over those tables. Inconclusive or invalid checks never override
the original selector. Gold SQL and labels are never used online.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

import sqlglot
from sqlglot import exp

from alphasql.algorithm.selection.execution_evidence import pool_identity
from alphasql.algorithm.selection.schema_signal import result_signature
from alphasql.llm_call.runtime import configure_environment


def _bounded_range(*args: int) -> range:
    result=range(*args)
    if len(result)>50:
        raise ValueError("range exceeds 50 values")
    return result


_SAFE_CALLS = {"len": len, "sum": sum, "min": min, "max": max,
               "sorted": sorted, "round": round, "abs": abs, "set": set,
               "list": list, "tuple": tuple, "int": int, "float": float,
               "str": str, "range": _bounded_range}
_SAFE_NODES = (
    ast.Expression, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.Name, ast.Load, ast.Store, ast.Subscript, ast.Slice,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.IfExp, ast.BoolOp,
    ast.Compare, ast.BinOp, ast.UnaryOp, ast.Call, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


def _json(text: str) -> dict[str, Any]:
    cleaned=text.strip()
    if cleaned.startswith("```"):
        cleaned=re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    value=json.loads(cleaned)
    if not isinstance(value,dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _safe_python(expression: str, tables: Mapping[str, list[dict[str, Any]]]) -> list[tuple[Any,...]]:
    if len(expression)>4000:
        raise ValueError("Python expression too long")
    tree=ast.parse(expression,mode="eval")
    nodes=list(ast.walk(tree))
    if len(nodes)>180 or any(not isinstance(node,_SAFE_NODES) for node in nodes):
        raise ValueError("Python expression uses an unsupported construct")
    if sum(len(node.generators) for node in nodes if isinstance(node,
           (ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)))>2:
        raise ValueError("Python expression has too many nested iterations")
    if sum(isinstance(node,ast.Mult) for node in nodes)>2:
        raise ValueError("Python expression has too many multiplications")
    for node in nodes:
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and abs(node.value)>100:
            raise ValueError("Numeric constant exceeds safe bound")
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and len(node.value)>200:
            raise ValueError("String constant exceeds safe bound")
        if isinstance(node,ast.Name) and (node.id.startswith("_") or
                                           node.id not in {"tables",*_SAFE_CALLS} and
                                           not any(isinstance(other,ast.Name) and
                                                   isinstance(other.ctx,ast.Store) and
                                                   other.id==node.id for other in nodes)):
            raise ValueError(f"Unsafe Python name: {node.id}")
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or
                                           node.func.id not in _SAFE_CALLS):
            raise ValueError("Only bounded safe builtins are allowed")
    environment={"__builtins__":{},"tables":tables,**_SAFE_CALLS}
    value=eval(compile(tree,"<restricted-solver>","eval"),environment,{})
    if isinstance(value,(str,int,float,bool)) or value is None:
        value=[(value,)]
    if not isinstance(value,(list,tuple)) or len(value)>50:
        raise ValueError("Solver must return at most 50 rows")
    return [tuple(row) if isinstance(row,(list,tuple)) else (row,) for row in value]


def _schema_slice(row: Mapping[str,Any], sqls: list[str]) -> dict[str,dict[str,str]]:
    tables={};referenced=set()
    for sql in sqls:
        parsed=sqlglot.parse_one(sql,read="sqlite")
        if not isinstance(parsed,(exp.Select,exp.Union,exp.Intersect,exp.Except)):
            raise ValueError("Only read-only SELECT candidates are supported")
        tables.update({node.name.lower():node.name for node in parsed.find_all(exp.Table)})
        referenced.update(node.name.lower() for node in parsed.find_all(exp.Column))
    if not tables or len(tables)>3:
        raise ValueError("Probe supports one to three referenced tables")
    full=row["full_schema"]["tables"]
    output={}
    for lower,name in tables.items():
        match=next((key for key in full if key.lower()==lower),None)
        if match is None:
            raise ValueError(f"Unknown table: {name}")
        columns=full[match]["columns"]
        selected={col:description.get("type") or "TEXT"
                  for col,description in columns.items()
                  if col.lower() in referenced}
        if not selected or len(selected)>30:
            raise ValueError("Unsupported column slice")
        output[match]=selected
    return output


def _validate_fixture(value: Mapping[str,Any], schema: Mapping[str,Mapping[str,str]]) -> dict[str,list[dict[str,Any]]]:
    tables=value.get("tables")
    if not isinstance(tables,dict):
        raise ValueError("Tester must provide tables")
    output={};total=0
    for table,columns in schema.items():
        rows=tables.get(table)
        if not isinstance(rows,list) or len(rows)>8:
            raise ValueError(f"Invalid fixture rows for {table}")
        cleaned=[]
        for row in rows:
            if not isinstance(row,dict) or any(col not in columns for col in row):
                raise ValueError(f"Unknown fixture column in {table}")
            if any(v is not None and not isinstance(v,(str,int,float,bool)) for v in row.values()):
                raise ValueError("Fixture values must be scalar")
            cleaned.append({col:row.get(col) for col in columns})
        output[table]=cleaned;total+=len(cleaned)
    if total>20:
        raise ValueError("Fixture exceeds 20 rows")
    return output


def _quote(identifier: str) -> str:
    return '"'+identifier.replace('"','""')+'"'


def _execute_fixture(schema: Mapping[str,Mapping[str,str]],
                     tables: Mapping[str,list[dict[str,Any]]], sql: str) -> list[tuple[Any,...]]:
    connection=sqlite3.connect(":memory:")
    try:
        for table,columns in schema.items():
            def affinity(kind: str) -> str:
                value=str(kind).upper()
                if "INT" in value: return "INTEGER"
                if any(word in value for word in ("REAL","FLOAT","DOUBLE")): return "REAL"
                if any(word in value for word in ("NUMERIC","DECIMAL")): return "NUMERIC"
                if "BLOB" in value: return "BLOB"
                return "TEXT"
            ddl=",".join(f"{_quote(col)} {affinity(kind)}" for col,kind in columns.items())
            connection.execute(f"CREATE TABLE {_quote(table)} ({ddl})")
            if tables[table]:
                names=list(columns)
                connection.executemany(
                    f"INSERT INTO {_quote(table)} ({','.join(map(_quote,names))}) VALUES ({','.join('?' for _ in names)})",
                    [[row.get(name) for name in names] for row in tables[table]])
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: 1,100000)
        cursor=connection.execute(sql)
        rows=cursor.fetchmany(51)
        if len(rows)>50:
            raise ValueError("SQL fixture result exceeds 50 rows")
        return [tuple(row) for row in rows]
    finally:
        connection.close()


def _request(client: Any, prompt: dict[str,Any], *, max_tokens: int) -> tuple[dict[str,Any],dict[str,int]]:
    response=client.chat.completions.create(
        model="Qwen3.8-27B",messages=[{"role":"user","content":json.dumps(prompt,ensure_ascii=False)}],
        temperature=0,max_tokens=max_tokens,
        extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    value=_json(response.choices[0].message.content or "")
    usage=response.usage
    return value,{"input_tokens":int(usage.prompt_tokens or 0) if usage else 0,
                  "output_tokens":int(usage.completion_tokens or 0) if usage else 0}


def eligible_pair(row: Mapping[str,Any], execution: Mapping[str,Any],
                  margin: float=.20) -> tuple[int,int,str] | None:
    candidates=row["rounds"][0]["candidates"]
    valid=sorted((i for i,c in enumerate(candidates) if c["execution"]["valid"]),
                 key=lambda i:(-candidates[i]["candidate_raw"],i))
    if len(valid)<2:
        return None
    first=valid[0]
    # Prefer a challenger from a different original-data result cluster.
    second=next((i for i in valid[1:] if execution["candidates"][i].get("result_hash") !=
                 execution["candidates"][first].get("result_hash")),valid[1])
    same_result=(execution["candidates"][first].get("result_hash") ==
                 execution["candidates"][second].get("result_hash"))
    if candidates[first]["candidate_raw"]-candidates[second]["candidate_raw"]>margin and not same_result:
        return None
    if candidates[first]["canonical_sql"]==candidates[second]["canonical_sql"]:
        return None
    return first,second,"same_result" if same_result else "close_scores"


def probe_one(client: Any,row: Mapping[str,Any],execution: Mapping[str,Any],
              pair: tuple[int,int,str]) -> dict[str,Any]:
    first,second,reason=pair
    sqls=[row["rounds"][0]["candidates"][i]["sql"] for i in (first,second)]
    schema=_schema_slice(row,sqls)
    tester_prompt={"task":"Create a tiny synthetic database that makes the two SELECT queries return different results. Return JSON only: {\"tables\":{table_name:[{column:value},...]}}. Use at most 8 rows per table and only listed columns. Respect the question and evidence, include boundary/null/unmatched rows when helpful. Do not decide which SQL is correct.",
                   "question":row["question"],"evidence":row.get("evidence") or "",
                   "schema":schema,"candidate_sql":sqls}
    fixture_data,first_usage=_request(client,tester_prompt,max_tokens=2048)
    tables=_validate_fixture(fixture_data,schema)
    outputs=[_execute_fixture(schema,tables,sql) for sql in sqls]
    if result_signature(outputs[0])==result_signature(outputs[1]):
        return {"status":"not_distinguishing","pair":[first,second],"reason":reason,
                "pool_sha256":pool_identity(row),"fixture":tables,"sql_outputs":outputs,
                "usage":first_usage,"llm_calls":1}
    solver_prompt={"task":"Solve the natural-language question on these fully visible tiny tables WITHOUT seeing any candidate SQL. Return JSON only: {\"python_expr\":\"...\"}. The expression must be a single safe Python expression using tables[table_name] (list of row dictionaries), comprehensions and builtins len,sum,min,max,sorted,round,abs,set,list,tuple,int,float,str,range. Return a list of result rows; use tuples for multi-column rows. No imports, functions, attributes, external data or SQL.",
                   "question":row["question"],"evidence":row.get("evidence") or "",
                   "schema":schema,"tables":tables}
    solver_data,second_usage=_request(client,solver_prompt,max_tokens=2048)
    expression=solver_data.get("python_expr")
    if not isinstance(expression,str):
        raise ValueError("Solver omitted python_expr")
    expected=_safe_python(expression,tables)
    matches=[result_signature(output)==result_signature(expected) for output in outputs]
    winner=([first,second][matches.index(True)] if sum(matches)==1 else None)
    column_counts=[len(row["rounds"][0]["candidates"][i]["execution"]["columns"])
                   for i in (first,second)]
    # A question may be ambiguous about projection shape. Treat a cross-shape
    # match as diagnostic evidence only, not permission to change Top-1.
    projection_ambiguity=column_counts[0]!=column_counts[1]
    return {"status":("projection_ambiguity" if projection_ambiguity and winner is not None
                      else "decisive" if winner is not None else "inconclusive"),
            "pair":[first,second],"reason":reason,"pool_sha256":pool_identity(row),
            "fixture":tables,"sql_outputs":outputs,"python_expr":expression,
            "python_output":expected,"matches":matches,
            "advisory_winner":winner,
            "recommended_index":None if projection_ambiguity else winner,
            "usage":{"input_tokens":first_usage["input_tokens"]+second_usage["input_tokens"],
                     "output_tokens":first_usage["output_tokens"]+second_usage["output_tokens"]},
            "llm_calls":2}


def collect(args: argparse.Namespace) -> None:
    configure_environment()
    import os
    from openai import OpenAI,DefaultHttpxClient
    if not os.environ.get("AIGC_API_KEY"):
        raise ValueError("AIGC_API_KEY is required")
    selected_ids=args.question_ids
    if args.split_file:
        if selected_ids:
            raise ValueError("Use either --question-ids or --split-file")
        selected_ids=json.loads(Path(args.split_file).read_text(encoding="utf-8"))["question_ids"][args.split]
    rows=([json.loads(Path(args.input,f'{qid}.json').read_text(encoding="utf-8"))
           for qid in selected_ids] if selected_ids else
          [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(args.input).glob('[0-9]*.json'))])
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    attempted=0;successful=0;eligible=0;failures=[]
    prior_count=len([path for path in output.glob("*.json")
                     if path.name!="collection_status.json"])
    with OpenAI(api_key=os.environ["AIGC_API_KEY"],
                base_url="https://aigc-api.hkust-gz.edu.cn/api/v2",
                http_client=DefaultHttpxClient(trust_env=False,timeout=90),
                max_retries=1) as client:
        for row in rows:
            path=output/f'{row["db_id"]}_{row["question_id"]}.json'
            if path.is_file():
                saved=json.loads(path.read_text(encoding="utf-8"))
                if saved["pool_sha256"]!=pool_identity(row):
                    raise ValueError(f"Frozen pool changed: {path}")
                continue
            stem=f'{row["db_id"]}_{row["question_id"]}.json'
            evidence=json.loads(Path(args.execution_dir,stem).read_text(encoding="utf-8"))
            pair=eligible_pair(row,evidence,args.margin)
            if pair is None:
                continue
            eligible+=1
            if prior_count+attempted>=args.max_pools:
                continue
            attempted+=1
            try:
                result=probe_one(client,row,evidence,pair)
                result.update(question_id=row["question_id"],db_id=row["db_id"])
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",
                                encoding="utf-8")
                successful+=1
                print(json.dumps({"question_id":row["question_id"],
                                  "status":result["status"],"winner":result.get("recommended_index")}),flush=True)
            except Exception as exc:
                failure={"question_id":row["question_id"],"db_id":row["db_id"],
                         "pool_sha256":pool_identity(row),"status":"probe_failed",
                         "type":type(exc).__name__,"error":str(exc)[:300],
                         "usage_status":"unavailable_for_failed_request"}
                path.write_text(json.dumps(failure,ensure_ascii=False,indent=2)+"\n",
                                encoding="utf-8")
                failures.append(failure)
                print(json.dumps(failures[-1]),flush=True)
    (output/"collection_status.json").write_text(json.dumps({"eligible":eligible,
        "attempted_this_run":attempted,"successful_this_run":successful,
        "failures":failures},indent=2)+"\n",encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True);p.add_argument("--execution-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--question-ids",type=int,nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split",choices=("train","validation","test"))
    p.add_argument("--margin",type=float,default=.20)
    p.add_argument("--max-pools",type=int,default=30)
    args=p.parse_args()
    if args.split_file and not args.split:
        p.error("--split is required with --split-file")
    collect(args)


if __name__=="__main__":
    main()

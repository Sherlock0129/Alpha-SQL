import fs from 'node:fs/promises';

const ROOT='D:/University/Alpha-SQL';
const RUN=ROOT+'/results/sql_selector_curriculum_p0a_seed101';
const FONT='Microsoft YaHei';
const C={ink:'#173047',body:'#263746',muted:'#62727E',blue:'#245F94',teal:'#287F78',red:'#AE4D3C',light:'#EEF3F6',white:'#FFFFFF'};
const ARMS=['U1','LP1','EA1','NoGroupEA1'];

function text(s,str,x,y,w,h,size=28,color=C.body,bold=false,name=''){
  const sh=s.shapes.add({geometry:'textbox',name:name||str.slice(0,24),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=str;
  sh.text.style={typeface:FONT,fontSize:size,color,bold,verticalAlignment:'top',alignment:'left',insets:0};
  return sh;
}
function title(s,str,n){
  s.background.fill=C.white;
  text(s,str,72,48,1136,74,44,C.ink,true,'slide-title');
  text(s,String(n).padStart(2,'0'),1160,674,48,26,18,C.muted,false,'page-number');
}
function foot(s,str){text(s,str,72,636,1080,45,20,C.muted,false,'source-caveat');}
function table(s,values,{x=72,y=170,w=1136,h=390,widths,fontsize=26,header=26}={}){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({fill:'#DCE4EA',width:1,style:'solid'});
  for(let r=0;r<values.length;r++)for(let col=0;col<values[0].length;col++){
    const cell=t.getCell(r,col);
    cell.fill=r===0?C.ink:r%2===0?'#F5F7F9':C.white;
    cell.text.style={typeface:FONT,fontSize:r===0?header:fontsize,bold:r===0,color:r===0?C.white:C.body,verticalAlignment:'middle'};
  }
  return t;
}
function notes(s,body,files=[]){
  s.speakerNotes.text=body+'\n\n本页依据：\n'+files.map(f=>ROOT+'/'+f).join('\n');
}

export async function buildDeck(p){
  p.theme.defaultFont=FONT;
  const summary=JSON.parse(await fs.readFile(RUN+'/paired_summary.json','utf8'));
  const manifest=JSON.parse(await fs.readFile(RUN+'/frozen_manifest.json','utf8'));
  const parity=JSON.parse(await fs.readFile(RUN+'/equivalence_report.json','utf8'));
  const diag=JSON.parse(await fs.readFile(RUN+'/diagnostics.json','utf8'));
  const probes={};
  for(const arm of ARMS)probes[arm]=(await fs.readFile(`${RUN}/${arm}/probe_trace.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
  if(!parity.passed||diag.ea_vs_lp.different_base_slots!==4)throw new Error('Unexpected source experiment, review slide narrative');
  const report=['reports/SQL_SELECTOR_CURRICULUM_P0A_RESULTS_ZH.md'];
  const artifact=['results/sql_selector_curriculum_p0a_seed101/paired_summary.json'];
  let s=p.slides.items[0];
  s.background.fill=C.ink;
  text(s,'固定训练集下的\nSQL 选择器错误感知课程学习',72,150,1120,210,54,C.white,true,'cover-title');
  text(s,'第一轮离线实验与等价性检查',76,390,1100,60,30,'#C5D6E2');
  text(s,'Dongx Xia\n2026年10月20日',76,552,800,80,25,C.white);
  notes(s,'这次汇报聚焦选择器的训练调度。目标是控制数据、模型和候选池，只研究错误类型是否能在训练动态和保持性回放之外提供额外信息。当前仅完成第一轮机制 pilot，尚没有性能提升结论。',report);

  s=p.slides.add();title(s,'研究问题与固定条件',2);
  text(s,'相同训练集下，错误语义能否帮助选择器更快达标？',72,167,1136,100,38,C.blue,true);
  text(s,'固定训练问题、候选 SQL、模型和总训练预算\n只改变训练样本的调度方式',72,292,1120,115,30);
  text(s,'比较条件：质量达到共同目标，同时控制新增回归\n新增回归：原自一致性选对，训练后的选择器反而选错',72,440,1120,112,28);
  foot(s,'正式达标门槛尚未冻结。本轮只检查实现、数据和调度机制。');
  notes(s,'研究目标不是简单提高训练集正确率，而是在固定问题集合下，缩短达到共同质量和回归约束的训练时间。这里的原选择器是冻结的执行自一致性。当前没有冻结质量和回归门槛，因此这一轮不能计算正式 time-to-target。',['reports/SQL_SELECTOR_CURRICULUM_EXPERIMENT_PROTOCOL_ZH.md',...report]);

  s=p.slides.add();title(s,'四组训练对照',3);
  table(s,[['方法','基础采样依据','保持性回放'],['U1','按问题均匀随机','相同'],['LP1','当前损失与学习进度','相同'],['EA1','LP 信号加错误组反馈','相同'],['EA 去掉语义项','EA 的 η=0，应与 LP1 等价','相同']],{y:160,h:355,widths:[265,650,221],fontsize:27});
  text(s,'每步 8 个 pair：6 个基础槽位 + 2 个回放槽位\n各组使用相同初始参数、回放序列和 anchor 序列',72,546,1136,80,26);
  foot(s,'“1”表示使用保持性回放。回放替换基础槽位，不额外增加 batch。');
  notes(s,'LP 关注 pair 的当前困难程度和近期损失下降。EA 在同一套 LP 控制器上增加错误组的误排率与进度反馈。四组的回放只来自 SC 正确且可组成正负 pair 的训练问题。η=0 的 EA 需要完整退化为 LP，避免实现差异污染研究归因。',['config/sql_selector_curriculum_protocol.yaml','alphasql/algorithm/selection/curriculum.py']);

  s=p.slides.add();title(s,'数据与训练设置',4);
  table(s,[['数据范围','问题数','候选 / pair'],['训练集','129','454 个候选'],['验证集','40','115 个候选'],['可形成正负 pair 的训练子集','20','82 个 pair'],['保持性回放子集','11','SC 正确的 pairwise 问题']],{y:158,w:1136,h:300,widths:[620,170,346],fontsize:25});
  text(s,'小型 MLP，48 个数值特征，保留自一致性信号\nseed 101，各组 250 步，前 25 步统一随机采样',72,493,1136,90,27);
  foot(s,'使用历史 direct 候选池。未生成新候选、未注入 gold，未处理 test 候选。');
  notes(s,`实际冻结的 train 为 ${manifest.train_questions} 题、${manifest.train_candidates} 个候选。正负 pair 只来自 ${manifest.train_pairwise_questions} 个问题，总共 ${manifest.train_pairs} 个。其余问题仍通过共同 pointwise anchor 参与训练。模型隐藏层64/32，dropout0.1，RankNet加0.2倍BCE，AdamW lr0.001。每10步共同probe，没有验证早停。分割按问题，不是跨数据库泛化评估。原始数据JSON容器会整体载入，但只对train/validation访问gold构造标签。`,['results/sql_selector_curriculum_p0a_seed101/frozen_manifest.json','results/sql_selector_curriculum_p0a_seed101/run_config.json']);

  s=p.slides.add();title(s,'η=0 与 LP1 的等价性检查通过',5);
  text(s,'0',72,170,260,180,148,C.teal,true,'zero-parameter-difference');
  text(s,'最终参数最大绝对差值',78,360,420,60,28,C.teal);
  text(s,'采样概率与完整抽样序列一致\n训练损失与梯度范数一致\n所有验证预测与 logits 一致\n最终模型参数逐位一致',565,190,640,270,30);
  text(s,'四组 warmup、保持性回放和 anchor 序列也一致',72,531,1136,66,28);
  foot(s,'9 项运行检查通过，23 项单元及回归测试通过。等价性成立不代表方法有效。');
  notes(s,'这一页说明对照实现成立。LP 与 NoGroupEA1 在有限250步轨迹中完全相同，包括最终tensor的bit-exact结果。单元测试还覆盖30轮随机probe的η=0概率等价性。这是有限轨迹检查，不是任意输入的数学证明。23项测试通过，实际用unittest运行，不是pytest。',['results/sql_selector_curriculum_p0a_seed101/equivalence_report.json',...report]);

  s=p.slides.add();title(s,'验证结果尚未改善',6);
  const resultRows=[['方法','正确题数','正确率','修复 / 新增回归'],['原自一致性','28 / 40','70.0%','参考基线']];
  for(const arm of ARMS){const v=summary[arm].final_validation;resultRows.push([arm==='NoGroupEA1'?'EA 去掉语义项':arm,`${v.correct} / ${v.questions}`,`${(v.ex*100).toFixed(1)}%`,`${v.fixes} / ${v.regressions}`]);}
  table(s,resultRows,{y:158,h:365,widths:[360,220,220,336],fontsize:27});
  text(s,'候选池中有正确 SQL 的问题共 31/40\n预算末四组均未修复 3 个 hard case，均新增 1 题回归',72,551,1136,78,27,C.red);
  foot(s,'采用现有执行签名匹配，非官方 BIRD EX。单 seed，未做显著性或非劣推断。');
  notes(s,'四组预算末结果均为27/40，SC为28/40。新增回归的条件比例是1/28=3.57%。候选覆盖31/40，所以有9题不可能由固定候选池的选择器修复。执行签名按行无序多重集比较，保留重复行和列顺序，浮点保留8位小数。没有使用官方BIRD evaluator，不能把它标为正式BIRD EX。',artifact);

  s=p.slides.add();title(s,'验证曲线始终没有超过自一致性',7);
  const colors={U1:C.blue,LP1:C.teal,EA1:C.red};
  const series=['U1','LP1','EA1'].map(arm=>({name:arm,values:probes[arm].map(r=>r.validation.correct),line:{fill:colors[arm],width:3,style:'solid'}}));
  series.push({name:'原自一致性',values:probes.U1.map(()=>28),line:{fill:'#929EAA',width:2,style:'dashed'}});
  const chart=s.charts.add('line',{position:{left:72,top:163,width:860,height:415},categories:probes.U1.map(r=>String(r.step)),series,hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontFamily:FONT,fontSize:21}},
    xAxis:{title:{text:'训练步数',textStyle:{fontFamily:FONT,fontSize:20}},textStyle:{fontFamily:FONT,fontSize:17},majorGridlines:null},
    yAxis:{min:24,max:32,majorUnit:2,title:{text:'验证正确题数 / 40',textStyle:{fontFamily:FONT,fontSize:20}},textStyle:{fontFamily:FONT,fontSize:20},numberFormatCode:'0',majorGridlines:{fill:'#E1E7EC',width:1,style:'solid'}},
    chartFill:C.white,plotAreaFill:C.white,dataLabels:{showValue:false},lineOptions:{smooth:false}});
  for(const style of [chart.legend?.textStyle,chart.dataLabels?.textStyle,chart.xAxis?.textStyle,chart.yAxis?.textStyle,chart.xAxis?.title?.textStyle,chart.yAxis?.title?.textStyle])if(style)style.typeface=FONT;
  text(s,'最高\n28/40',968,215,230,135,44,C.ink,true);
  text(s,'训练集预算末\n100/129\n达到候选 oracle',968,392,240,150,27,C.body);
  foot(s,'纵轴截取 24–32 题。η=0 曲线与 LP1 重合，图中省略。最高值未用于挑选模型。');
  notes(s,'这张图使用全部26个checkpoint。三个模型曲线重叠较多，表示选择差异只涉及一两道问题，不能用线条看起来不同暗示明显增益。所有组验证最高只有28题，没有超过SC。训练集最后均达到100/129的候选oracle，U1首次在第80步probe达到，LP和EA在70步达到。训练拟合提早不代表验证达标或实际净加速。',['results/sql_selector_curriculum_p0a_seed101/U1/probe_trace.jsonl','results/sql_selector_curriculum_p0a_seed101/LP1/probe_trace.jsonl','results/sql_selector_curriculum_p0a_seed101/EA1/probe_trace.jsonl']);

  s=p.slides.add();title(s,'错误语义对采样的实际影响很小',8);
  text(s,'4 / 1500',72,152,630,100,70,C.blue,true);
  text(s,'EA 与 LP 不同的基础采样槽位，约 0.27%',72,261,1120,58,28);
  table(s,[['弱结构组','pair 数','不同训练问题'],['投影 / 去重','78','19'],['过滤 / literal','56','17'],['表与 Join','65','18'],['聚合 / 排序等计算','48','13']],{y:346,h:245,widths:[620,220,296],fontsize:25});
  foot(s,'分组允许重叠，不能相加。unknown 组为空。最大分布总变差为 0.001765。');
  notes(s,'EA的语义项确实参与了权重计算，但总体调度变化很小。最大单pair概率差是0.000349629，最大总变差0.001765242。所有组只覆盖同一批20个训练问题，弱结构组高度重叠，归一化question-equivalent约3.99到7.00，κ=10的收缩使反馈接近全局均值。共有常数项在指数归一化后抵消，这可能解释语义项作用较弱，但不是已经验证的因果结论。当前标签对alias和等价表达式敏感，未经过人工审计。',['results/sql_selector_curriculum_p0a_seed101/diagnostics.json','results/sql_selector_curriculum_p0a_seed101/weak_pair_labels.json']);

  s=p.slides.add();title(s,'逐题分析：输出完整性与值对齐',9);
  table(s,[['问题','当前结果','候选之间的关键差异'],['1616','新增回归','first name 与 last name 两列顺序交换\n需先核查评价口径'],['8005','未修复','只输出美国大学数量\n遗漏符合条件的大学总数'],['8896','未修复',"literal 用了 'baking product'\n正候选使用 'baking products'"],['8902','未修复','只输出总时长\n遗漏问题要求的 recipe_id']],{y:158,h:420,widths:[130,225,781],fontsize:26});
  text(s,'另有 9 题候选池本身缺少正确 SQL，选择器无法修复',72,598,1136,40,27,C.muted);
  notes(s,'这些是根据冻结候选SQL核查的差异。1616两个候选都返回同一个客户的姓名，当前签名对列序敏感，不能直接把这一题当成业务语义完全错误。8005要求两个计数，8896是数据库literal单复数，8902要求recipe标识和总时长。这些例子提示仅使用结构统计和schema覆盖可能不够，但本轮没有特征因果消融。逐题明细见error_analysis.csv，共四组160行。',['results/sql_selector_curriculum_p0a_seed101/error_analysis.csv','results/sql_selector_curriculum_p0a_seed101/frozen_records.json']);

  s=p.slides.add();title(s,'下一步先核查标签，再扩大比较',10);
  table(s,[['下一步工作','需要得到的证据'],['审计训练 pair 的错误标签','排除 alias、引号和等价 SQL 的伪差异\n确认组别对应可解释的 SQL 错误'],['检查 train-only 调度反馈','语义项是否提供不同于 loss / progress 的信息\n是否产生合理、足够的采样差异'],['重新冻结实验设置','统一训练标签、模型、预算和推理规则\n不按本次验证结果搜索 η'],['进入多 seed 比较','先冻结共同质量与回归门槛\n计入 probe 和调度成本，比较稳定达标时间']],{y:160,h:440,widths:[380,756],fontsize:25});
  foot(s,'当前不直接扩大成 36 runs。若更换模型表示，所有调度组同时更换并独立评估。');
  notes(s,'下一步优先做训练标签审计和训练信号诊断，不用验证结果反向调大语义系数。等语义项能产生合理异质性以后，再运行多seed主比较。质量和回归门槛需要来自独立基线pilot，正式比较应加上去掉不必要probe的production-cost对照。若另外增加question-aware表示，必须给所有调度组同样的模型，不能将模型升级收益记到课程上。',report);

  s=p.slides.add();title(s,'目前的结论与讨论',11);
  text(s,'实验对照已经跑通\n错误语义的额外收益尚未观察到',72,171,1120,145,42,C.ink,true);
  text(s,'标签粒度\n哪些差异足以定义训练调度中的“错误类型”？',72,365,1120,94,29);
  text(s,'模型与调度的边界\n当前数值表示是否足以支持语义课程的检验？',72,495,1120,94,29);
  foot(s,'最终要回答：在共同质量和回归约束下，错误语义是否能缩短训练时间。');
  notes(s,'想请老师或师兄讨论两点。第一，适合调度的错误分组应细到什么程度，哪些差异只是SQL写法而不是错误。第二，当前数值MLP是否已经足以检验这个想法，还是需要先独立验证question-aware表示。现阶段证据支持实现检查通过，不支持性能、加速或不增加回归的结论。',report);

  s=p.slides.add();title(s,'附录：运行成本与结果边界',12);
  const costs=[['阶段 / 方法','测得耗时','解释'],['共同数据准备','6.05 秒','读取缓存、SQLite 执行与冻结'],['共同 CUDA 初始化','0.265 秒','独立记录，不归入 U1']];
  for(const arm of ARMS)costs.push([arm==='NoGroupEA1'?'EA 去掉语义项':arm,summary[arm].cost.wall_seconds.toFixed(3)+' 秒','训练、probe、checkpoint 与日志']);
  table(s,costs,{y:154,h:390,widths:[360,220,556],fontsize:24});
  text(s,'单次运行很短，顺序与首次训练栈开销会影响耗时\n尚未冻结达标阈值，本轮不报告训练加速倍数',72,574,1136,70,26,C.red);
  notes(s,'这些耗时只用于审计预算。U1是首组，首次优化器和训练栈启动可能影响耗时，约1秒的微型运行不适合比较现实速度。当前没有quality_tau和regression_r_max，所以不报告正式time-to-target。完整实验无外部模型API调用。',['results/sql_selector_curriculum_p0a_seed101/frozen_manifest.json','results/sql_selector_curriculum_p0a_seed101/run_config.json',...artifact]);
  return {slides:p.slides.items.length,font:FONT,sourceRun:RUN};
}

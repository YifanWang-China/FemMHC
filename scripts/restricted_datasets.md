# 受限数据下载说明

以下数据不能由脚本代替研究者接受数据使用协议，也不应把账号密码写进仓库。

## mcPHASES

1. 登录 PhysioNet。
2. 完成 Credentialed Health Data Use Agreement 要求的身份认证和培训。
3. 在数据页申请访问：`https://physionet.org/content/mcphases/1.0.0/`
4. 获批后按页面给出的认证下载命令保存到：
   `<dataset-root>\restricted\mcphases_1.0.0`

## GLOBEM

1. 登录 PhysioNet。
2. 完成所需 CITI 课程和 DUA。
3. 在数据页申请访问：`https://physionet.org/content/globem/1.1/`
4. 获批后保存到：
   `<dataset-root>\restricted\globem_1.1`

## TILES-2018

在 `https://tiles-data.isi.edu/dataset2018_details` 提交对应的学术或商业 DUA。该数据约100 GB，商业DUA明确限制为内部研究且禁止商业使用。保存到：
`<dataset-root>\restricted\tiles_2018`

## All of Us Fitbit

通过 `https://www.researchallofus.org/register/` 由合格机构申请 Researcher Workbench。数据和训练作业必须留在工作台；不要设计本地下载流程。

## BUMP

项目页：`https://www.synapse.org/Synapse:syn25953345`

需成为 Synapse certified/qualified user，并提交 intended-use statement 和 IRB-approved protocol。获批后保存到：
`<dataset-root>\restricted\bump_syn25953345`

## SWAN Core + SWAN Sleep/Actigraphy

- NIA AgingResearchBiobank：`https://agingresearchbiobank.nia.nih.gov/studies/swan/details`
- ICPSR SWAN系列：`https://www.icpsr.umich.edu/web/ICPSR/series/253`

NIA数据申请材料明确写有 `No Commercial Use`。早期SWAN Sleep Study包含约370名中年女性、最长约35天腕部actigraphy/睡眠日记；Visit 15 actigraphy子研究约1,196名女性有可用数据。获批后保存到：
`<dataset-root>\restricted\swan`

## 安全约定

- 凭证只通过交互式登录、系统凭证存储或临时环境变量传入。
- 不把密码、API token、cookie或DUA表单保存到项目目录。
- 受限数据与公开数据物理分目录。
- 任何基于受限数据训练的权重都标记来源和允许用途，不能自动进入产品发布流水线。

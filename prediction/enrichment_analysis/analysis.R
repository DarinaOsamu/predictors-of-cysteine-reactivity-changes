#加载R包
library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)

#文件读取
data <- read.table(file="in_only_EnID.txt",header=FALSE)
data$V1 <- as.character(data$V1) #需要character格式
#groupGO分析
#ggo <- groupGO(gene = data$V1, OrgDb = org.Hs.eg.db, ont = "CC",level = 3,readable = TRUE)
#barplot(ggo,drop=TRUE,showCategory=10)
#enrichGO分析
ego_ALL <- enrichGO(gene = data$V1,
                    universe = names(geneList), #背景基因集
                    OrgDb = org.Hs.eg.db, #没有organism="human"，改为OrgDb=org.Hs.eg.db
                    ont = "ALL", #也可以是 CC  BP  MF中的一种
                    pAdjustMethod = "BH", #矫正方式 holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”中的一种
                    pvalueCutoff = 1, #P值会过滤掉很多，可以全部输出
                    qvalueCutoff = 1,
                    readable = TRUE) #Gene ID 转成gene Symbol ，易读
head(ego_ALL,2)
dotplot(ego_ALL,title="EnrichmentGO_ALL_dot")
barplot(ego_ALL, showCategory=10,title="EnrichmentGO_ALL_bar")

ego_CC <- enrichGO(gene = data$V1,
                    universe = names(geneList), #背景基因集
                    OrgDb = org.Hs.eg.db, #没有organism="human"，改为OrgDb=org.Hs.eg.db
                    ont = "CC", #也可以是 CC  BP  MF中的一种
                    pAdjustMethod = "BH", #矫正方式 holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”中的一种
                    pvalueCutoff = 1, #P值会过滤掉很多，可以全部输出
                    qvalueCutoff = 1,
                    readable = TRUE) #Gene ID 转成gene Symbol ，易读
head(ego_CC,2)
dotplot(ego_CC,title="EnrichmentGO_CC_dot")
barplot(ego_CC, showCategory=10,title="EnrichmentGO_CC_bar")

ego_BP <- enrichGO(gene = data$V1,
                   universe = names(geneList), #背景基因集
                   OrgDb = org.Hs.eg.db, #没有organism="human"，改为OrgDb=org.Hs.eg.db
                   ont = "BP", #也可以是 CC  BP  MF中的一种
                   pAdjustMethod = "BH", #矫正方式 holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”中的一种
                   pvalueCutoff = 1, #P值会过滤掉很多，可以全部输出
                   qvalueCutoff = 1,
                   readable = TRUE) #Gene ID 转成gene Symbol ，易读
head(ego_BP,2)
dotplot(ego_BP,title="EnrichmentGO_BP_dot")
barplot(ego_BP, showCategory=10,title="EnrichmentGO_BP_bar")

ego_MF <- enrichGO(gene = data$V1,
                   universe = names(geneList), #背景基因集
                   OrgDb = org.Hs.eg.db, #没有organism="human"，改为OrgDb=org.Hs.eg.db
                   ont = "MF", #也可以是 CC  BP  MF中的一种
                   pAdjustMethod = "BH", #矫正方式 holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”中的一种
                   pvalueCutoff = 1, #P值会过滤掉很多，可以全部输出
                   qvalueCutoff = 1,
                   readable = TRUE) #Gene ID 转成gene Symbol ，易读
head(ego_MF,2)
dotplot(ego_MF,title="EnrichmentGO_MF_dot")
barplot(ego_MF, showCategory=10,title="EnrichmentGO_MF_bar")

#enrichKEGG分析
ekegg_ALL <- enrichKEGG(gene = data$V1,
                        organism="hsa",
                        pvalueCutoff = 1
                        )
dotplot(ekegg_ALL,title="EnrichmentKEGG_ALL_dot")
barplot(ekegg_ALL, showCategory=10,title="EnrichmentKEGG_ALL_bar")
#enrichDO分析
edo_ALL <- enrichDO(gene = data$V1,
                    ont = "DO",
                    pvalueCutoff = 1,
                    qvalueCutoff = 1
                    )
dotplot(edo_ALL,title="EnrichmentDO_ALL_dot")
barplot(edo_ALL, showCategory=10,title="EnrichmentDO_ALL_bar")


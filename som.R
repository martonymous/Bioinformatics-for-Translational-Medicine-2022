# B4TM, Self-Organizing Maps (SOM), Code

# Note: the packages kohonen, tidyr, dplyr and readr are required.

# Load libraries
library(kohonen) # used for SOM
library(tidyr)
library(dplyr)
library(readr)

# Read train data files
X_all <- read_tsv("data/train_call.txt") # all the features
X_SHAP <- read_csv("data/selected_SHAP.csv") # top 25 slected with SHAP
X_chi2 <- read_csv("data/selected_chi2.csv") # top 25 selected with Chi2

# Read class labels data
labels <- read.table("data/train_clinical.txt", sep = "\t", header=T)

# Remove the columns: Chromosome, start, end, nclone for X_all
X_all <- select(X_all,-Chromosome,-Start,-End,-Nclone)

# Format the X_all to join it with the class labels file later
array_id <- colnames(X_all)
X_all_1 <- as_tibble(t(X_all))
X_all <- as_tibble(t(X_all))
X_all$Sample <- array_id
colnames(X_chi2)[1] <- "Sample"

# Join the train and class label data for the SOM for the two sub-types as we 
# want to filter them according to the subtype later
table_all <- full_join(X_all, labels)
table_chi2 <- full_join(X_chi2, labels)

# Filter HER2+ for the all the features (X_all) and the Chi2 selected (X_Chi2)
table_all <- table_all %>% filter(Subgroup != "HER2+") %>% 
  select(-Sample,-Subgroup)
table_chi2 <- table_chi2 %>% filter(Subgroup != "HER2+") %>% 
  select(-Sample,-Subgroup)

# Filter HER2+ class labels for the two sub-type maps
labels_two_sub <- labels %>% filter(Subgroup != "HER2+")

### Self-organizing maps (SOM) ###

# Change the labels into numbers for the three sub-type maps
labels[labels=='HER2+'] <- 1 # change HER2 to 0
labels[labels=='HR+'] <- 2 # change HR+ to 1
labels[labels=='Triple Neg'] <- 3 # change TP to 2

# Change the labels into numbers for the two sub-type maps
labels_two_sub[labels_two_sub=='HR+'] <- 1
labels_two_sub[labels_two_sub=='Triple Neg'] <- 2

# We use a 5x5 grid for the maps
c1 = 5
c2 = 5

## SOM for the three subtypes, without feature selection ##
kohmap_all <- som(as.matrix(X_all_1), grid = somgrid(c1,c2, "hexagonal"), 
                  rlen = 100)

# Mapping plot
plot(kohmap_all, type = "mapping",col=c('red','green','blue')[as.integer(labels[,2])], pchs=c(15,16,17)[as.integer(labels[,2])],
     main = "SOM without feature selection", shape="straight")
# Create the legend
labels_names <- c('HER2+', 'HR+', 'Triple Neg')
legend(6,4,pch=15:17, col=c('red','green','blue'), 
       legend=as.character(labels_names), cex=1, y.intersp = 0.1, x.intersp = 0.1,
       bty = "n")

## SOM for the three subtypes, with SHAP feature selection ##
kohmap_shap <- som(as.matrix(X_SHAP[,2:26]), grid = somgrid(c1,c2, "hexagonal"), 
                  rlen = 100)

# Mapping plot
plot(kohmap_shap, type = "mapping",col=c('red','green','blue')[as.integer(labels[,2])], pchs=c(15,16,17)[as.integer(labels[,2])],
     main = "SOM with SHAP feature selection", shape="straight")
# Create the legend
labels_names <- c('HER2+', 'HR+', 'Triple Neg')
legend(6,4,pch=15:17, col=c('red','green','blue'), 
       legend=as.character(labels_names), cex=1, y.intersp = 0.1, x.intersp = 0.1,
       bty = "n")

## SOM for the two subtypes, without feature selection ##
kohmap_all_two <- som(as.matrix(table_all), grid = somgrid(c1,c2, "hexagonal"), 
                   rlen = 100)

# Mapping plot
plot(kohmap_shap, type = "mapping",col=c('green','blue')[as.integer(labels_two_sub[,2])], 
     pchs=c(15,16,17)[as.integer(labels_two_sub[,2])],
     main = "SOM without feautre selection", shape="straight")
# Create the legend
labels_names <- c('HR+', 'Triple Neg')
legend(6,4,pch=15:17, col=c('red','green','blue'), 
       legend=as.character(labels_names), cex=1, y.intersp = 0.1, x.intersp = 0.1,
       bty = "n")

## SOM for the two subtypes, with Chi2 feature selection ##
kohmap_chi2 <- som(as.matrix(table_chi2), grid = somgrid(c1,c2, "hexagonal"), 
                      rlen = 100)

# Mapping plot
plot(kohmap_shap, type = "mapping",col=c('green','blue')[as.integer(labels[,2])], 
     pchs=c(15,16,17)[as.integer(labels_two_sub[,2])],
     main = "SOM Chi2 selected features", shape="straight")
# Create the legend
labels_names <- c('HR+', 'Triple Neg')
legend(6,4,pch=15:17, col=c('green','blue'), 
       legend=as.character(labels_names), cex=1, y.intersp = 0.1, x.intersp = 0.1,
       bty = "n")

## End of code ## 


#hw 6 code
#enter precode:
library(tidymodels)
library(tidyverse)
library(dplyr)
#library(corrr)
library(ggplot2)
library(discrim)
library(klaR)
library(glmnet)
library(ISLR)
#install.packages('rpart.plot')
library(rpart.plot)
#install.packages('vip')
library(vip)
library(randomForest)
#install.packages('xgboost')
library(xgboost)
#install.packages('ranger')
library(ranger)
tidymodels_prefer()
pokemon <- read.csv("data/Pokemon.csv")
set.seed(1234)

#EXERCISE 1
library(janitor)
pokemon <- pokemon %>%
  clean_names()
#filter out rarer classes
pokemon_filter <- pokemon %>%
  filter(type_1 == c("Bug", "Fire", "Grass", "Normal", "Water", "Psychic"))
#convert type_1 and legendary to factors
pokemon_factor <- pokemon_filter %>%
  mutate(type_1 = factor(type_1), 
         legendary = factor(legendary))
#initial_split with percentage; stratify type_1
pokemon_split <- initial_split(pokemon_factor, 
                               prop = 0.80, 
                               strata = type_1)
pokemon_split

pokemon_train <- training(pokemon_split)
pokemon_test <- testing(pokemon_split)
#v-fold with v=5; strata = type_1
pokemon_folds <- vfold_cv(pokemon_train, v = 5, 
                          strata = type_1)
pokemon_folds
#recipe
pokemon_recipe <-
  recipe(type_1 ~ legendary + generation + sp_atk + attack + speed + defense + 
           hp + sp_def, data = pokemon_train) %>%
  step_dummy(c("legendary", "generation")) %>% #all_nominal_predictors?
  step_center(all_predictors()) %>% #or empty it out
  step_scale(all_predictors())
pokemon_recipe

#EXERCISE 2: correlation matrix
library(corrplot)
pokemon_train %>%
  select_if(is.numeric) %>%
  cor(use = "complete.obs") %>%
  corrplot(type = "lower", diag = FALSE, method = "color")

#EXERCISE 3: decision tree
tree_spec <- decision_tree() %>%
  set_engine("rpart")
class_tree_spec <- tree_spec %>%
  set_mode("classification")

class_tree_wf <- workflow() %>%
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_formula(type_1 ~ legendary + generation + sp_atk + attack + 
                speed + defense + hp + sp_def) #recipe?

param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

tune_res <- tune_grid(
  class_tree_wf, 
  resamples = pokemon_folds, 
  grid = param_grid, 
  metrics = metric_set(roc_auc)
)

autoplot(tune_res)

#EXERCISE 4: roc_auc
collect_metrics(tune_res)
best_penalty <- select_best(tune_res, metric = "roc_auc") #rsq??

tree_final <- finalize_workflow(class_tree_wf, best_penalty)
#do i need to change workflow in hw5??

tree_final_fit <- fit(tree_final, data = pokemon_train)
roc_auc1 <- augment(tree_final_fit, new_data = pokemon_test) %>%
  select(type_1, starts_with(".pred")) %>%
  roc_auc(type_1, .pred_Bug:.pred_Water) #0.531

#EXERCISE 5: rpart.plot
tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot()

#EXERCISE 5: random forest model and wrkflow
bagging_spec <- rand_forest(mtry = tune(), trees = tune(), 
                            min_n = tune()) %>%
  set_engine("ranger", importance = 'impurity') %>%
  set_mode("classification")

class_tree_wf2 <- workflow() %>%
  add_model(bagging_spec) %>%
  add_recipe(pokemon_recipe)
param_grid2 <- grid_regular(mtry(range = c(1, 8)), trees(range = c(10, 2000)),
                            min_n(range = c(1, 8)), levels = 8)

#EXERCISE 6: roc_auc as metric -- takes a few minutes to run
tune_res2 <- tune_grid(
  class_tree_wf2, 
  resamples = pokemon_folds, 
  grid = param_grid2, 
  metrics = metric_set(roc_auc)
)
autoplot(tune_res2)

#EXERCISE 7: roc_auc of best performing model
collect_metrics(tune_res2)
best_penalty2 <- select_best(tune_res2, metric = "roc_auc")

tree_final2 <- finalize_workflow(class_tree_wf2, best_penalty2)
#do i need to change workflow in hw5??

tree_final_fit2 <- fit(tree_final2, data = pokemon_train)
roc_auc2 <- augment(tree_final_fit2, new_data = pokemon_test) %>%
  select(type_1, starts_with(".pred")) %>%
  roc_auc(type_1, .pred_Bug:.pred_Water) #0.653

#EXERCISE 8: vip()
vip(extract_fit_engine(tree_final_fit2))

#EXERCISE 9: boosted tree model
boost_spec <- boost_tree(trees = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
class_tree_wf3 <- workflow() %>%
  add_model(boost_spec) %>%
  add_recipe(pokemon_recipe)
param_grid3 <- grid_regular(trees(range = c(10, 2000)),levels = 10)
tune_res3 <- tune_grid(
  class_tree_wf3, 
  resamples = pokemon_folds, 
  grid = param_grid3, 
  metrics = metric_set(roc_auc)
)
autoplot(tune_res3)

collect_metrics(tune_res3)
best_penalty3 <- select_best(tune_res3, metric = "roc_auc")

tree_final3 <- finalize_workflow(class_tree_wf3, best_penalty3)
#do i need to change workflow in hw5??

tree_final_fit3 <- fit(tree_final3, data = pokemon_train)
roc_auc3 <- augment(tree_final_fit3, new_data = pokemon_test) %>%
  select(type_1, starts_with(".pred")) %>%
  roc_auc(type_1, .pred_Bug:.pred_Water) #0.558

#EXERCISE 10
result <- bind_rows(roc_auc1, roc_auc2, roc_auc3) %>%
  tibble() %>%
  mutate(model = c('pruned tree model', 'random forest model', 
                   'boost tree model'), 
         .before = .metric)
#random forest did best:
collect_metrics(tune_res2)
final_penalty <- select_best(tune_res2)

final_tree <- finalize_workflow(class_tree_wf2, final_penalty)
#do i need to change workflow in hw5??

final_fit <- fit(final_tree, data = pokemon_test)
final_auc_roc <- augment(final_fit, new_data = pokemon_test) %>%
  select(type_1, starts_with(".pred")) %>%
  roc_auc(type_1, .pred_Bug:.pred_Water) #.967

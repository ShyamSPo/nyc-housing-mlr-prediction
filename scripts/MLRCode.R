# ============================================================
# NYC Housing Price - Model 1: Multiple Linear Regression
# (Updated with Log-Transformed Price)
# ============================================================

library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(MASS)

install.packages("gridExtra")
library(gridExtra)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATA
# ------------------------------------------------------------
housing_clean <- read.csv("NY-House-Cleaned.csv", stringsAsFactors = FALSE)

cat("Loaded dataset:", nrow(housing_clean), "rows,", ncol(housing_clean), "cols\n")


# ------------------------------------------------------------
# 2. ADDITIONAL CLEANUP SPECIFIC TO MODELING
# ------------------------------------------------------------

# Drop columns with no predictive value for price
housing_model <- housing_clean %>%
  dplyr::select(-STATE, -MAIN_ADDRESS, -FORMATTED_ADDRESS, -SUBLOCALITY, -PRICE_PER_SQFT)

# Standardize LOCALITY to the 5 NYC boroughs
housing_model <- housing_model %>%
  mutate(BOROUGH = case_when(
    LOCALITY %in% c("Manhattan", "New York", "New York County")  ~ "Manhattan",
    LOCALITY %in% c("Brooklyn", "Kings County", "Flatbush")      ~ "Brooklyn",
    LOCALITY %in% c("Queens", "Queens County")                   ~ "Queens",
    LOCALITY %in% c("The Bronx", "Bronx County")                 ~ "Bronx",
    LOCALITY %in% c("Staten Island", "Richmond County")          ~ "Staten Island",
    TRUE ~ "Other"
  )) %>%
  dplyr::select(-LOCALITY)

# Remove "Other" borough rows
housing_model <- housing_model %>%
  filter(BOROUGH != "Other")

cat("Rows after borough cleanup:", nrow(housing_model), "\n")

# Consolidate rare TYPE categories into "Other"
type_counts <- table(housing_model$TYPE)
rare_types  <- names(type_counts[type_counts < 30])

housing_model <- housing_model %>%
  mutate(TYPE = ifelse(TYPE %in% rare_types, "Other", TYPE))

# Encode categorical variables as factors
housing_model <- housing_model %>%
  mutate(
    TYPE    = as.factor(TYPE),
    BOROUGH = as.factor(BOROUGH)
  )

cat("\nProperty types used in model:\n")
print(table(housing_model$TYPE))

cat("\nBoroughs used in model:\n")
print(table(housing_model$BOROUGH))


# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (70/30)
# ------------------------------------------------------------
set.seed(42)

train_index <- createDataPartition(housing_model$PRICE, p = 0.70, list = FALSE)
train_data  <- housing_model[train_index, ]
test_data   <- housing_model[-train_index, ]

cat("\nTrain size:", nrow(train_data), "| Test size:", nrow(test_data), "\n")


# ------------------------------------------------------------
# 4. FIT MULTIPLE LINEAR REGRESSION WITH LOG(PRICE)
#    Log-transforming price compresses the right skew from
#    luxury listings and stabilizes variance across price ranges
# ------------------------------------------------------------
mlr_model <- lm(log(PRICE) ~ BEDS + BATH + PROPERTYSQFT + LATITUDE + LONGITUDE +
                  TYPE + BOROUGH,
                data = train_data)

cat("\n--- Full MLR Model Summary (log scale) ---\n")
print(summary(mlr_model))


# ------------------------------------------------------------
# 5. STEPWISE VARIABLE SELECTION VIA AIC
# ------------------------------------------------------------
mlr_stepwise <- stepAIC(mlr_model, direction = "both", trace = FALSE)

cat("\n--- Stepwise Selected Model Summary (log scale) ---\n")
print(summary(mlr_stepwise))


# ------------------------------------------------------------
# 6. PREDICTIONS ON TEST SET
# ------------------------------------------------------------
log_predictions  <- predict(mlr_stepwise, newdata = test_data)
test_predictions <- exp(log_predictions)


# ------------------------------------------------------------
# 7. MODEL EVALUATION (in original dollar terms)
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7. MODEL EVALUATION
# ------------------------------------------------------------

# --- Log Scale Evaluation (for comparison with other models) ---
log_test_actual <- log(test_data$PRICE)

rmse_log    <- sqrt(mean((log_test_actual - log_predictions)^2))
ss_res_log  <- sum((log_test_actual - log_predictions)^2)
ss_tot_log  <- sum((log_test_actual - mean(log_test_actual))^2)
r_squared_log <- 1 - (ss_res_log / ss_tot_log)

cat("\n--- MLR Evaluation (Log Scale) ---\n")
cat("Log RMSE:      ", round(rmse_log, 3), "\n")
cat("Log R-Squared: ", round(r_squared_log, 4), "\n")

# --- Dollar Scale Evaluation ---
rmse      <- sqrt(mean((test_data$PRICE - test_predictions)^2))
mae       <- mean(abs(test_data$PRICE - test_predictions))
ss_res    <- sum((test_data$PRICE - test_predictions)^2)
ss_tot    <- sum((test_data$PRICE - mean(test_data$PRICE))^2)
r_squared <- 1 - (ss_res / ss_tot)

cat("\n--- MLR Evaluation (Dollar Terms) ---\n")
cat("RMSE:      $", round(rmse, 2), "\n")
cat("MAE:       $", round(mae, 2),  "\n")
cat("R-Squared: ",  round(r_squared, 4), "\n")


# ------------------------------------------------------------
# 8. VISUALIZATIONS
# ------------------------------------------------------------

# --- Plot 1: Actual vs Predicted Prices ---
actual_vs_predicted <- data.frame(
  Actual    = test_data$PRICE,
  Predicted = test_predictions
)

ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "MLR: Actual vs Predicted Housing Prices",
    subtitle = "Red dashed line = perfect prediction",
    x        = "Actual Price ($)",
    y        = "Predicted Price ($)"
  ) +
  theme_minimal()

ggsave("mlr_actual_vs_predicted.png", width = 8, height = 6)


# --- Plot 2: Residuals vs Fitted (log scale) ---
residuals_df <- data.frame(
  Fitted    = fitted(mlr_stepwise),
  Residuals = residuals(mlr_stepwise)
)

ggplot(residuals_df, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title    = "MLR: Residuals vs Fitted Values (Log Scale)",
    subtitle = "Should appear as random scatter around zero after log transform",
    x        = "Fitted Values (log scale)",
    y        = "Residuals (log scale)"
  ) +
  theme_minimal()

ggsave("mlr_residuals_vs_fitted.png", width = 8, height = 6)


# --- Plot 3: Average Price by Borough ---
borough_avg_price <- housing_model %>%
  group_by(BOROUGH) %>%
  summarise(Avg_Price = mean(PRICE)) %>%
  arrange(desc(Avg_Price))

ggplot(borough_avg_price, aes(x = reorder(BOROUGH, Avg_Price), y = Avg_Price, fill = BOROUGH)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = scales::comma) +
  coord_flip() +
  labs(
    title = "Average Housing Price by Borough",
    x     = "Borough",
    y     = "Average Price ($)"
  ) +
  theme_minimal()

ggsave("mlr_avg_price_by_borough.png", width = 8, height = 6)

# ---------------- Other Plots --------------------
housing_eda <- housing_clean %>%
  mutate(BOROUGH = case_when(
    LOCALITY %in% c("Manhattan", "New York", "New York County")  ~ "Manhattan",
    LOCALITY %in% c("Brooklyn", "Kings County", "Flatbush")      ~ "Brooklyn",
    LOCALITY %in% c("Queens", "Queens County")                   ~ "Queens",
    LOCALITY %in% c("The Bronx", "Bronx County")                 ~ "Bronx",
    LOCALITY %in% c("Staten Island", "Richmond County")          ~ "Staten Island",
    TRUE ~ "Other"
  )) %>%
  filter(BOROUGH != "Other") %>%
  mutate(LOG_PRICE = log(PRICE))

# --- Plot 4: Price vs Square Footage by Borough ---
ggplot(housing_eda, aes(x = PROPERTYSQFT, y = LOG_PRICE, color = BOROUGH)) +
  geom_point(alpha = 0.4, size = 1.0) +
  geom_smooth(method = "lm", se = FALSE, color = "red",
              linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(labels = scales::comma) +
  facet_wrap(~ BOROUGH, nrow = 1) +
  labs(
    title    = "Property Square Footage vs Log Price by Borough",
    subtitle = "Red dashed line = linear trend per borough",
    x        = "Property Square Footage",
    y        = "Log Price"
  ) +
  theme_minimal() +
  theme(
    plot.title    = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("eda_price_vs_sqft.png", width = 14, height = 5)
cat("Saved: eda_price_vs_sqft.png\n")

# --- Plot 5: Price Distribution ---
# Raw price distribution
plot_raw_price <- ggplot(housing_eda, aes(x = PRICE)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_continuous(labels = scales::dollar_format(scale = 1e-6, suffix = "M")) +
  labs(
    title = "Distribution of Housing Prices",
    x     = "Price",
    y     = "Number of Properties"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Log transformed price distribution
plot_log_price <- ggplot(housing_eda, aes(x = LOG_PRICE)) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(
    title = "Distribution of Log Housing Prices",
    x     = "Log Price",
    y     = "Number of Properties"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Save side by side
png("eda_price_distribution.png", width = 1200, height = 500)
gridExtra::grid.arrange(plot_raw_price, plot_log_price, ncol = 2)
dev.off()

cat("Saved: eda_price_distribution.png\n")

# ------------------------------------------------------------
# 9. SAVE RESULTS SUMMARY
# ------------------------------------------------------------
mlr_results <- data.frame(
  Model     = "Multiple Linear Regression",
  RMSE      = round(rmse, 2),
  MAE       = round(mae, 2),
  R_Squared = round(r_squared, 4)
)

write.csv(mlr_results, "mlr_evaluation_results.csv", row.names = FALSE)

cat("\nMLR complete. Plots and results saved.\n")
cat("Objects available: mlr_model, mlr_stepwise, train_data, test_data\n")


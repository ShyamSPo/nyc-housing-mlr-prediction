# ============================================================
# NYC Housing Price - Data Preprocessing
# ============================================================

library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
house_data <- read.csv("NY-House-Dataset.csv", stringsAsFactors = FALSE)

cat("Original dimensions:", nrow(house_data), "rows,", ncol(house_data), "cols\n")

# ------------------------------------------------------------
# 2. DROP COLUMNS THAT WON'T HELP MODELS
#    Removing address/text fields and redundant location columns
# ------------------------------------------------------------
house_data <- house_data %>%
  select(-BROKERTITLE,          # broker name, irrelevant to price
         -ADDRESS,              # raw address string
         -STREET_NAME,          # too granular
         -LONG_NAME,            # property name, not useful
         -ADMINISTRATIVE_AREA_LEVEL_2)  # redundant with SUBLOCALITY


# ------------------------------------------------------------
# 3. REMOVE MISSING VALUES
# ------------------------------------------------------------
cat("Missing values before removal:\n")
print(colSums(is.na(house_data)))

house_data <- house_data %>% drop_na()

cat("Rows after removing NAs:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 4. REMOVE DUPLICATES
# ------------------------------------------------------------
house_data <- house_data %>% distinct()
cat("Rows after removing duplicates:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 5. FIX DATA TYPES
#    PRICE, BEDS, BATH, PROPERTYSQFT should all be numeric
# ------------------------------------------------------------
house_data <- house_data %>%
  mutate(
    PRICE       = as.numeric(PRICE),
    BEDS        = as.numeric(BEDS),
    BATH        = as.numeric(BATH),
    PROPERTYSQFT = as.numeric(PROPERTYSQFT)
  )

# Drop any rows that became NA after coercion
house_data <- house_data %>% drop_na()
cat("Rows after type coercion cleanup:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 6. REMOVE PRICE OUTLIERS
#    NYC has extreme luxury listings that will distort models.
#    We remove the bottom 1% (likely data errors) and 
#    top 1% (ultra-luxury outliers).
#    We also remove any listings with PRICE <= 0.
# ------------------------------------------------------------
price_low  <- quantile(house_data$PRICE, 0.01)
price_high <- quantile(house_data$PRICE, 0.99)

cat("Price range before outlier removal: $", min(house_data$PRICE), 
    "to $", max(house_data$PRICE), "\n")

house_data <- house_data %>%
  filter(PRICE > 0,
         PRICE >= price_low,
         PRICE <= price_high)

cat("Price range after outlier removal: $", min(house_data$PRICE), 
    "to $", max(house_data$PRICE), "\n")
cat("Rows after price outlier removal:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 7. REMOVE PROPERTYSQFT OUTLIERS
#    Same logic — remove bottom and top 1%
#    Also remove any zero or negative sqft values
# ------------------------------------------------------------
sqft_low  <- quantile(house_data$PROPERTYSQFT, 0.01)
sqft_high <- quantile(house_data$PROPERTYSQFT, 0.99)

house_data <- house_data %>%
  filter(PROPERTYSQFT > 0,
         PROPERTYSQFT >= sqft_low,
         PROPERTYSQFT <= sqft_high)

cat("Rows after sqft outlier removal:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 8. FILTER UNREALISTIC BEDS / BATH VALUES
#    Remove listings with 0 beds/baths or extreme counts
# ------------------------------------------------------------
house_data <- house_data %>%
  filter(BEDS >= 0,
         BATH >= 0,
         BEDS <= 20,
         BATH <= 20)

cat("Rows after beds/bath filter:", nrow(house_data), "\n")

# ------------------------------------------------------------
# 9. CATEGORICAL VARIABLES
#    TYPE     -> factor (property type: Condo, House, etc.)
#    LOCALITY -> factor (borough-level: Manhattan, Brooklyn, etc.)
#    SUBLOCALITY -> factor (neighborhood level)
# ------------------------------------------------------------
house_data <- house_data %>%
  mutate(
    TYPE        = as.factor(TYPE),
    LOCALITY    = as.factor(LOCALITY),
    SUBLOCALITY = as.factor(SUBLOCALITY)
  )

cat("\nProperty types in data:\n")
print(table(house_data$TYPE))

cat("\nLocalities (boroughs) in data:\n")
print(table(house_data$LOCALITY))

# ------------------------------------------------------------
# 10. ADD FEATURE: PRICE PER SQFT
#     Useful derived variable for feature importance analysis
# ------------------------------------------------------------
house_data <- house_data %>%
  mutate(PRICE_PER_SQFT = PRICE / PROPERTYSQFT)

# ------------------------------------------------------------
# 11. TRAIN / TEST SPLIT (70/30)
#     Set seed for reproducibility
# ------------------------------------------------------------
set.seed(42)

train_index <- createDataPartition(house_data$PRICE, p = 0.70, list = FALSE)

train_data <- house_data[train_index, ]
test_data  <- house_data[-train_index, ]

cat("\nTrain set size:", nrow(train_data), "\n")
cat("Test set size: ", nrow(test_data),  "\n")

# ------------------------------------------------------------
# 12. SUMMARY 
# ------------------------------------------------------------
cat("\n--- Final Cleaned Dataset Summary ---\n")
print(summary(house_data[, c("PRICE", "BEDS", "BATH", "PROPERTYSQFT", 
                             "PRICE_PER_SQFT", "LATITUDE", "LONGITUDE")]))

cat("\nPreprocessing complete. Objects ready: house_data, train_data, test_data\n")

# ------------------------------------------------------------
# 13. EXPORT CLEANED DATA TO CSV
# ------------------------------------------------------------
write.csv(house_data, "NY-House-Cleaned.csv", row.names = FALSE)
cat("Cleaned dataset exported to NY-House-Cleaned.csv\n")

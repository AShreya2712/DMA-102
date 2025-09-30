names(data) <- c("PM25","PM10","NO2","CO","SO2","O3","Temperature","Humidity","WindSpeed","AQI")
data[] <- lapply(data, as.numeric)
x <- as.matrix(data %>% select(-AQI))
y <- data$AQI
nn_formula <- AQI ~ PM25 + PM10 + NO2 + CO + SO2 + O3 + Temperature + Humidity + WindSpeed
nn_model <- neuralnet(nn_formula, data = data, hidden = c(2,1), linear.output = TRUE)
plot(nn_model)
nn_pred <- compute(nn_model, data %>% select(-AQI))$net.result
nn_rmse <- sqrt(mean((y - nn_pred)^2))
cat("Neural Network RMSE:", nn_rmse, "\n")
dall <- xgb.DMatrix(data = x, label = y)

xgb_model <- xgboost(data = dall,
                     nrounds = 20,       
                     objective = "reg:squarederror",
                     max_depth = 2,     
                     eta = 0.1,
                     verbose = 0)
xgb_pred <- predict(xgb_model, dall)
xgb_rmse <- sqrt(mean((y - xgb_pred)^2))
cat("XGBoost RMSE:", xgb_rmse, "\n")
lm_model <- lm(AQI ~ PM25 + PM10 + NO2 + CO + SO2 + O3 + Temperature + Humidity + WindSpeed, data = data)
lm_pred <- predict(lm_model, newdata = data)
lm_rmse <- sqrt(mean((y - lm_pred)^2))
cat("Linear Regression RMSE:", lm_rmse, "\n")
results <- data.frame(
  Actual = y,
  NN_Predicted = nn_pred,
  XGB_Predicted = xgb_pred,
  LM_Predicted = lm_pred
)

ggplot(results, aes(x = Actual)) +
  geom_point(aes(y = NN_Predicted, color = "Neural Network"), size = 3) +
  geom_point(aes(y = XGB_Predicted, color = "XGBoost"), size = 3) +
  geom_point(aes(y = LM_Predicted, color = "Linear Regression"), size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Actual vs Predicted AQI",
       y = "Predicted AQI", x = "Actual AQI") +
  scale_color_manual(values = c("Neural Network"="blue", "XGBoost"="red", "Linear Regression"="green")) +
  theme_minimal()
rmse_table <- data.frame(
  Model = c("Neural Network", "XGBoost", "Linear Regression"),
  RMSE = c(nn_rmse, xgb_rmse, lm_rmse)
)

print(rmse_table)

flags = flags(
  flag_numeric("nodes", 64),
  flag_numeric("batch_size", 32),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("dropout_rate", 0.7) 
)

model = keras_model_sequential()
model %>%
  layer_dense(units = flags$nodes, activation = "relu", input_shape = dim(training_data_scaled)[2]) %>%
  layer_dropout(rate = flags$dropout_rate) %>%
  layer_dense(units = flags$nodes, activation = "relu") %>%
  layer_dropout(rate = flags$dropout_rate) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_adam(lr = flags$learning_rate),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(
  train_matrix, train_labels,
  epochs = 20,
  batch_size = flags$batch_size,
  validation_data = list(validation_matrix, val_labels)
)



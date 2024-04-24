# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
    "growth_rate": 32,
    "block_config": (6, 12, 24, 16),
    "num_init_features": 64,
    "bn_size": 4,
    "drop_rate": 0,
    "num_classes": 10
	# ...
}

training_configs = {
	"learning_rate": 0.01,
    "epochs": 30,
    "learning_rate": 0.01,
    "train_ratio": 0.8
	# ...
}

### END CODE HERE
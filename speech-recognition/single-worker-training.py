import base

"""
Let's start the training process.
"""

batch_size = 32
dataset, num_to_char, char_to_num = base.ctc_dataset(batch_size)
model = base.build_and_compile_ctc_model(num_to_char, char_to_num)

# Define the number of epochs.
epochs = 1
# Callback function to check transcription on the val set.
# Train the model
history = model.fit(dataset, epochs=epochs)

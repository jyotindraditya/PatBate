import tensorflow as tf
import numpy as np

with open("american_psycho.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Text length:", len(text))
print(text[:500])


vocab = sorted(set(text))
vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)

char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}
encoded_text = np.array([char2idx[c] for c in text])
SEQ_LENGTH = 100

inputs = []
targets = []

for i in range(len(encoded_text) - SEQ_LENGTH):
    inputs.append(encoded_text[i:i + SEQ_LENGTH])
    targets.append(encoded_text[i + 1:i + SEQ_LENGTH + 1])

inputs = np.array(inputs)
targets = np.array(targets)

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(
    inputs,
    targets,
    epochs=10,
    batch_size=64
)

def generate_text(model, start_string, num_chars=500, temperature=1.0):
    input_indices = [char2idx[c] for c in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    generated_text = start_string

    for _ in range(num_chars):
        predictions = model(input_indices)
        predictions = predictions[:, -1, :] / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        generated_text += idx2char[predicted_id]

        input_indices = tf.concat(
            [input_indices, tf.expand_dims([predicted_id], 0)],
            axis=1
        )
    return generated_text


print(generate_text(model, start_string="There is an idea", num_chars=500))

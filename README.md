<h1>Music Transformer</h1>

<p>This is an attempt at simplifying the Transformer architecture to the point where it could be comfortably taught to undergrad CS students.

The big assumption that my simplification is based on is that, since matmulling across the sequence dimension does not give sufficient positional information to make state-of-the-art predictions, then we can remove any transfer of information along the sequence dimension <i>except</i> the positional encoding inside the attention modules.

So I removed the fully connected sublayers from the encoder and decoder modules.

The second, much smaller assumption is that the original paper put the LayerNorm in the wrong place, and had to do a lot of gymnastics during training to make up for that fact.

To remedy this, I simply put the LayerNorm as the first part of each sublayer. I also take the LayerNorm along the word dimension instead of the sequence dimension. I have no justification for doing this, and in fact have no justification for the LayerNorm being in the model at all. I think we can probably get rid of it, but I needed to rush out a working architecture, so I left it in.

Finally, I move the positional encoding into the Attention calculation, like in TransformerXL. This is based on the assumption that you cannot predict how words/pitches are related along the sequence dimension without having first figured out which words/pitches are related at all (imagine trying to figure out the rhythm of a song you haven't heard yet).</p>

![Image of simplified architecture](https://github.com/MacKenzieHnC/MusicTransformer.jl/blob/main/architecture/MyTransformerArchitecture.png?raw=true)

<h2>Example</h2>

```Julia
using MusicTransformer

# Data size
word_size = 10
sequence_length = 20
num_instruments = 4

# Model size
N = 6
head_count = 8
latent_size = 32


song = rand(word_size, sequence_length, num_instruments)

model = Transformer(N, head_count, latent_size, word_size)

prediction = model(song, get_positional_data(song))
```

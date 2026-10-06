# ML reading group: Creativity

These papers are my own code implementations of papers from Yixin's reading group (and thus the original credit and models go to those authors; this is just for playing around with their ideas).

I was interested in implementing and improving the "Alien Sampling" paper because:
- it seemed arbitrary in some ways. Why is _conceptual_ content the only thing that apparently matters in an artwork? Van Gogh didn't just think "sunflowers" and that was it. I have not yet addressed this satisfactorily within the context of this implementation.
- Why use GPT-2 for a problem of essentially inferring edges in a graph of only a couple thousand nodes ('concepts')? I replaced this with naive bayes. It's easier for me.
- I wanted to practice using LLMs and I ran some locally on a laptop.

I wanted to implement the Creator-Appraiser framework because I think it gets closer to the heart of what meaningful ideation and generation is about. For chemical design for instance, we should make decisions in generation that fit our desired outcomes.

_Each subdirectory has a more in-depth description of the contents of that implementation. A lot of credit to Claude, Gemini, ChatGPT for building most of the code, and helping with my personal research discovery process related to this project._

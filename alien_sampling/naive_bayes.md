Let $S=\{c_0,\ldots, c_7\}$. We want $c_8$. That, is we want to find the ninth concept to complete a group of 8.

Estimate

$$P(v∣S)\propto P(S\cup \{v\})$$

For example,

$$P(S\cup \{v\}) \approxeq \prod_{i\in S} P(v∣i)$$

where

$P(v∣i)=\frac{\#{edges containing i}}{\#{edges containing both i,v}}.$

This is analogous to Naive Bayes.
# Guiding document for MNIST Creator/Appraiser

The appraiser as autoencoder actually should have its full encoder frozen, and only *the first layer of the decoder* should be unfrozen.

After appraiser warmup, the first decoder layer is something fixed--say phi0.

Creator is warmed up or loaded from pre-saved checkpoint. creator_warmup_checkpoint is saved if not already.

Appraiser is warmed up or loaded from pre-saved checkpoint. appraiser_warmup_checkpoint is saved if not already.

The Creator learns by a loop:
1. The appraiser receives a creation c from the creator.
2. The appraiser is set back to its initial setting phi0.
3. We record l0, the initial cross-entropy loss of the appraiser evaluated on c.
4. Inner training loop (appraiser learning): T steps, default T=10
    a. The appraiser first layer decoder weights are updated to minimize the cross entropy loss.
5. We record lT, the loss after step T. Creator reward is R=(l0-lT) * exp(-(l0-mu)^2/tau^2) * exp(-lT/tau), and creator loss is negative the reward. mu defaults to 0.08, tau default value will be for now equal to mu.
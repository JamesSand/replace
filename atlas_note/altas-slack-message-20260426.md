Done. it’s under “theory_new.tex” https://www.overleaf.com/project/6939f773b97259ae1fffb15c @频道
[晚上 9:56]@Hanqing Zhu @Zhizhou Sha @Sagnik I want each of you to read it independently, and DM me your feedback privately (not this channel; I hate public echo)
[晚上 9:56]Notes on what I did:
[晚上 9:58]I gave three old claims under attack

RLVR is nearly isospectral, dominated by ambient transport, and that ISO explicitly enforces this geometry to improve RLVR
I instead defined a new diagnostic geometric suite, using classical matrix perturbation tools, but in somehow newly oganized way
[晚上 9:58]The old version looked more ambitious, but its ambition was brittle:

current asymptotic theorem should be downgraded because the proof only supports a cumulative finite-horizon bound, not vanishing relative drift from the stated sublinear condition
exact exponential spectral decay should be demoted because the real conceptual need is a nontrivial gap, not an exact exponential spectrum.
[晚上 9:59]I believe we should not try to prove that RLVR is necessarily isospectral. We do not have that theorem. We also should not try to prove ISO must beat AdamW or Muon. not have that theorem either.
[晚上 10:00]So what is presented now? (Hoffman-Wielandt-Mirsky, Weyl, Wedin, orthogonal projection identities, and first-order singular-value perturbation are standard tools)

the exact checkpoint-level model-class comparison is paper-specific. The theorem compares the best source-subspace fit, left-only hybrid, right-only hybrid, and target-subspace transport on the same truncated checkpoint transition. This is substantially stronger than the old Procrustes-only “inner mixing fails” baseline because the source-subspace model is allowed to choose the best inner matrix, not just a rigid Procrustes rotation
Corollary 3.3 gives a theorem-matched evaluation protocoI. It forces Section 4.2 to report basis-invariant quantities instead of visually persuasive but potentially gauge-sensitive singular-vector plots
Proposition 4.1 gives a clean local reason ISO is coherent: the diagonal core diag(U^T G V) is exactly the first-order singular-value drift direction, and fixed-spectrum optimization removes that coordinate. This is not a proof of superiority, but it is a better mechanism than “Adam explores noise”
[晚上 10:00]It is NOT a new theory proving why RLVR works but it’s safe
[晚上 10:02]Paper logic-wise: (1) we shall still motivate from experiments RLVR in our studied settings shows small spectral drift and nontrivial low-rank subspace movement. (2) next we formalize gauge-invariant checkpoint diagnostics that separate source-subspace fitting from target-subspace transport. (3) final product - ISO is a fixed-spectrum optimizer aligned with this observed geometry, and its value is demonstrated empirically.
[晚上 10:02]How this new theory supports ISO? Plain language:
[晚上 10:03]
The theory mainly says: IF useful RLVR movement mostly occurs through singular subspaces and not singular values (which itself is experimental!!!!), then a fixed-spectrum optimizer is a principled parameterization. Proposition 4.1 makes this precise by identifying the first-order spectrum-changing coordinates
Be clear: it does not prove the benefit of ISO! That must only come from experiments. This distinction matters because ICML reviewers already raised the exact concern: if AdamW is already nearly isospectral, why does freezing the spectrum help? Reviewer zjn4 explicitly asked for direct evidence that freezing the spectrum improves motivation, plus hyperparameter sweeps, wall-clock costs, and clearer algorithm details..
（已编辑）
[晚上 10:04]The remaning concern:

theory itself is not too exciting. true, but it’s correct & it links to practice. what else you want?
“The truncation rank is arbitrary.” This is serious attack. You must fix with rank sensitivity, retained energy, and boundary-gap reporting - read my note!
atlaswang  [晚上 10:04]
I also change the title to “ISO: An Isospectral Stiefel Optimizer for RLVR Post-Training” @Hanqing Zhu
atlaswang  [晚上 10:05]
Contribution wise, let us say (abs + intro) something like: “We introduce ISO, to our knowledge the first fixed-spectrum Stiefel-factor optimizer designed specifically for RLVR post-training”? (I want to claim the first RLVR-native optimizer… but too broad??)
atlaswang  [晚上 10:05]
回复了一个消息列:or “ISO: Isospectral Stiefel Optimization for RLVR”
atlaswang  [晚上 10:06]
回复了一个消息列:for safer positioning of “first”, say something like:

ISO is an RLVR-native optimizer in the sense that its parameterization is derived from the empirical weight-space geometry of RLVR, rather than inherited from pre-training or SFT.
[晚上 10:07]read my theory note, I believe it’s the lead authors’ job to integrate it in main paper? I don’t want to work in your messy construction site @此处 （已编辑） 
atlaswang  [晚上 10:25]
I can see it still takes lots of work to integrate the theory to writeup story. highly nontrivial work
[晚上 10:25]first make sure you fully know what I composed …
atlaswang  [晚上 11:01]
okay I think I need clarify more:

Frankly: the new theory is not intrinsically specific to RLVR as mathematics. It is specific to RLVR only through the empirical regime it is used to diagnose and the optimizer design it motivates
Theorem 2.3 is true for any matrix path. Wedin stability is true for any matrix perturbation. The checkpoint reconstruction theorem works for any two checkpoints. The fixed-spectrum tangent proposition works for any matrix with simple singular values.
So why we need this suite?? The theory is RLVR-specific in application, not RLVR-specific in theorem assumptions. It becomes RLVR-specific only when paired with the empirical claims of the paper:
In the studied reasoning RLVR runs, singular values barely move, while SFT produces spectral reshaping.
Replacing the RL-trained spectrum with the base spectrum causes little or no performance loss.
Training only the spectrum does not produce meaningful RLVR improvement.
Checkpoint transitions are better fit by target-subspace transport than by source-subspace fitting.
ISO is designed by freezing exactly the degrees of freedom that appear empirically inactive or risky in RLVR.

Those are RLVR-specific empirical facts. The theory then formalizes what follows conditional on that empirical regime

If the RLVR checkpoints exhibit small spectral drift, then these are the mathematically correct gauge-invariant diagnostics for deciding whether the observed movement is source-subspace fitting, one-sided transport, or target-subspace transport.
So what we actually did  is to develop a matrix-geometric diagnostic framework for the empirical isospectral-transport regime observed in reasoning RLVR.
The theory formalizes the consequences of the observed near-isospectrality and gives gauge-invariant tests for whether RLVR checkpoint motion is better explained by source-subspace fitting or target-subspace transport.
[晚上 11:02]I know the theory part can read confusing but PLEASE READ. Because later you have to pose all these correct in draft …
atlaswang  [凌晨 12:05]
those are quantites Section 4.2 should actually report
Screenshot 2026-04-26 at 1.05.06 AM.png atlaswang  [凌晨 12:34]
@频道 I did another full audit. I can confirm the math is def. right
[凌晨 12:35]the remaining barrier is YOU MUST UNDERSTAND WHAT I WROTE and then agree/disagree with me. @Sagnik @Zhizhou Sha @Hanqing Zhu: the same - please READ, and communicate with me via DM. ASK ANY QUESTION YOU HAVE ASAP!
[凌晨 12:35]because we have to converge on storyline at any minute now, it has to do with both experiments / abalations and massive rewriting!
atlaswang  [凌晨 12:35]
It would be fair, internally, to say that almost all of the mathematics is textbook matrix perturbation theory or a simple composition of standard linear-algebra facts
atlaswang  [凌晨 12:36]
so we must preempt this by being honest
[凌晨 12:37]Let me remind again: the true interesting is the combined story:
[凌晨 12:37]
RLVR checkpoints empirically show small spectral drift compared with SFT
[凌晨 12:37]2. Sigma replacement does not destroy performance
[凌晨 12:37]3. Sigma-only training fails or underperforms
[凌晨 12:37]4. Checkpoint transitions are better fit by target-subspace transport than by source-subspace fitting
[凌晨 12:37]5. ISO operationalizes this geometry by freezing the spectrum and optimizing Stiefel factors.
[凌晨 12:39]When you rewrite contributions, your top-3 shall be:
[凌晨 12:40]
Empirical discovery: reasoning RLVR appears near-isospectral in the studied settings.
Diagnostic framework: checkpoint transitions are better explained by target-subspace transport than source-subspace fitting.
Optimizer: ISO exploits this geometry and improves stability or time-to-target.
The theory should support points 2 and 3. Most what I wrote shall go Appendix!
atlaswang  [凌晨 12:40]
ANY ATTEMPT you try to over-sell theory will be ATTACKED! @Hanqing Zhu this is like your apollo paper, you don’t win by theorems here.
Hanqing Zhu  [凌晨 1:17]
回复了一个消息列:I take a look and agree most of things but only one thing:

trace-constraint ablation is FAKE, as zhzihou's previous exps has wrong loss impl? and cannot be reproduced iirc 
[凌晨 1:17]so we may cannot use this as the optimizer motivation
atlaswang  [凌晨 1:21]
@Zhizhou Sha confirm?
Hanqing Zhu  [凌晨 1:36]
and i am more than agree with the central claim
The defensible claim is: relative to a general Euclidean update, a fixed-spectrum parameterization removes explicit first-order singular- value drift coordinates, and the paper’s empirical results indicate that doing so is helpful in the
studied RLVR regime. 

by doing in the ISO way we should see a better convergence, maybe in longer run, we may still reach a same upper limit of performance (which is bounded by base model and data, not optimization itself) （已编辑）
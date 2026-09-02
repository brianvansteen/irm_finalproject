**Pre-registration.** These predictions were finalised and committed
before any scoring was run. Finalised on 2026-08-28.

## Notation

|Symbol|Meaning|
|-|-|
|**↓↓**|targeted criterion expected to be the largest negative mover of all 6 criteria|
|↓|targeted criterion expected to fall as a side-effect|
|~|targeted criterion expected to be roughly unchanged (small positive)|

**Note on "unchanged".** The six attribution shares sum to 1, therefore the six deltas sum to zero. Therefore, if one share falls, the others **must** rise. Unchanged "~" therefore
means "small positive, roughly in proportion to its baseline share", and no cell can be zero.

---

## Predicted Change (attribution share, modified − base)

|Modification|num. accuracy|completeness|clarity|succinctness|transparency|depth|
|-|-|-|-|-|-|-|
|**corrupt_figure**|**↓↓**|~|↓|~|~|~|
|**drop_input**|~|**↓↓**|↓|~|↓|~|
|**reverse_steps**|↓|~|**↓↓**|~|↓|~|
|**add_padding**|~|~|↓|**↓↓**|~|~|
|**remove_derivation**|↓|↓|~|~|**↓↓**|↓|
|**append_overdepth**|~|~|~|↓|~|**↓↓**|

## Predicted reward change

All six modifications should lower the total reward. This is a separate check from the attribution deltas and is not subject to the sum-to-zero constraint, so the reward can fall freely. A modification that shifts attribution without lowering reward would be a notable negative result.

The expected ordering of severity:

- corrupt_figure and remove_derivation should produce the largest reward drops, since they directly damage what the answer **is**

- add_padding and append_overdepth should produce the smallest reward drops, since they add rather than remove.

---

## Reasoning for each predicted side-effect

**corrupt_figure > clarity ↓.** The final figure no longer follows from the previous steps. Clarity includes internal consistency, so conclusions that contradict the calculations is a clarity failure mode.

**drop_input > transparency ↓.** A step now uses a figure the reader was never given, so the derivation cannot be fully reconstructed. Transparency requires that inputs be named, and one of them no longer is.

**reverse_steps > transparency ↓.** The steps still name their inputs and operations, but not logically sequential, so reconstruction is unlikely.

**add_padding > clarity ↓.** Restatement and boilerplate lengthen the text without changing content. This is the weakest of the predicted couplings, and clarity should be unaffected, since the derivation is untouched. If this cell comes out flat, that is a **correct** prediction.

**remove_derivation > completeness ↓ and depth ↓.** This is the most entangled modification. Removing the derivation removes the named inputs (reduced transparency), but it also removes the required information (reduced completeness) and leaves an answer shallower than the question warrants (reduced depth).

**append_overdepth > succinctness ↓.** The additional analysis is both mismatched in depth (the target) and superfluous to the question (succinctness). These two criteria are structurally similar and this modification is difficult to separate . If the target-hit rate for this row is low because succinctness moved further, that is an interpretable result about criterion overlap, not a failure of attribution.

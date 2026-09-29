# Shared Activity Taxonomy — Mapping Decision

To run any cross-dataset experiment (train on X, test on Y), every dataset's
labels must be mapped onto one common set of activity names. This document
records what each dataset actually contains, where the four datasets agree
and disagree, and which shared taxonomy this project uses and why. This is
a real methodological decision, not busywork — get it wrong (e.g. silently
dropping mismatched classes) and the cross-dataset numbers in Phase 4 are
meaningless.

## 1. Native label sets

| Dataset  | Native activities (as labelled by original authors) |
|----------|-------------------------------------------------------|
| UCI-HAR  | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| WISDM    | Walking, Jogging, Upstairs, Downstairs, Sitting, Standing |
| PAMAP2   | lying, sitting, standing, walking, running, cycling, nordic_walking, watching_tv, computer_work, car_driving, ascending_stairs, descending_stairs, vacuum_cleaning, ironing, folding_laundry, house_cleaning, playing_soccer, rope_jumping |
| MHEALTH  | standing_still, sitting_relaxing, lying_down, walking, climbing_stairs, waist_bends_forward, frontal_elevation_arms, knees_bending, cycling, jogging, running, jump_front_back |

PAMAP2 and MHEALTH both include several activities no other dataset has
(ironing, driving, waist bends, jumping jacks, etc.) — these are dropped
entirely for cross-dataset experiments, since a class that exists in the
target but never appears in the source is untestable by definition.

## 2. Where the four datasets actually agree

Going class-by-class (see full derivation in the loader `ACTIVITY_MAP`
dicts):

| Shared class | UCI-HAR | WISDM | PAMAP2 | MHEALTH | Coverage |
|---|---|---|---|---|---|
| **WALKING**  | ✅ WALKING | ✅ Walking | ✅ walking | ✅ walking | 4 / 4 |
| **SITTING**  | ✅ SITTING | ✅ Sitting | ✅ sitting | ✅ sitting_relaxing | 4 / 4 |
| **STANDING** | ✅ STANDING | ✅ Standing | ✅ standing | ✅ standing_still | 4 / 4 |
| **STAIRS*** | ✅ UP+DOWN | ✅ Up+Down | ✅ Asc+Desc | ✅ climbing_stairs | 4 / 4 (after merge) |
| **LYING**   | ✅ LAYING | ❌ absent | ✅ lying | ✅ lying_down | 3 / 4 |
| **RUNNING**  | ❌ absent | ✅ Jogging | ✅ running | ✅ jogging + running | 3 / 4 |
| **CYCLING**  | ❌ absent | ❌ absent | ✅ cycling | ✅ cycling | 2 / 4 |

\* UCI-HAR, WISDM and PAMAP2 all distinguish ascending vs. descending
stairs; MHEALTH's `climbing_stairs` label does not specify direction. To
keep STAIRS usable across all four datasets, up/down are **merged into a
single STAIRS class everywhere**, including in UCI-HAR/WISDM/PAMAP2 where
direction info exists but has to be discarded for consistency. (Direction
is retained as a secondary label internally, so within-dataset experiments
can still test up/down separately if useful later.)

## 3. Decision: two-tier taxonomy

**Tier 1 — Core-4 (used for the full 4-dataset × 4-dataset cross-dataset
matrix, Phase 4):**

```
WALKING, SITTING, STANDING, STAIRS
```

All four datasets support these four classes after the stairs-direction
merge, so this is the taxonomy used for every cross-dataset pair in the
main generalization-gap experiment and for the Deep CORAL adaptation
experiments.

**Tier 2 — Extended-5 (used only for the 3-dataset subset UCI-HAR /
PAMAP2 / MHEALTH, as a secondary/bonus experiment):**

```
WALKING, SITTING, STANDING, STAIRS, LYING
```

WISDM is excluded from any experiment using this taxonomy since it has no
lying/resting class. This tier is optional — include it if time permits
in Semester 2 as an extension showing the gap holds even with a richer
label set, but the Core-4 taxonomy is the one all headline results use.

RUNNING and CYCLING are **not** included in either tier: RUNNING would
require dropping UCI-HAR (which has neither jogging nor running), and
CYCLING would require dropping both UCI-HAR and WISDM — at that point
only 2 datasets remain, which defeats the purpose of a cross-dataset
study. They're left as a possible "stretch" experiment on the 2-dataset
PAMAP2/MHEALTH pair only, not a core deliverable.

## 4. Note for the report

This exact situation — needing to shrink from each dataset's native
activity set down to a smaller common core — is not unique to this
project; the DAGHAR benchmark paper (Napoli et al., 2024, Scientific
Data) made the identical call for the same reason (to include WISDM and
UCI-HAR, which have fewer native classes than the others), settling on
4–6 shared activities depending on which datasets were combined. Citing
that precedent in the report is a legitimate way to pre-empt the
"why did you drop N classes" question in a viva.

## 5. Practical implementation note

The label-remapping logic (native label → Tier-1/Tier-2 shared label)
should live in one shared module (`src/loaders/taxonomy.py`, to be
written alongside the preprocessing pipeline in Phase 2) rather than
being duplicated inside each per-dataset loader, so there is a single
place to audit and update the mapping as the project progresses.

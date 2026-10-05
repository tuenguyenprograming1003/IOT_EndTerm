# Diagnostic: classifier test accuracy (LOSO, seed 0) for log-mel variants

Reason: with the specified S = log1p(P) on float audio in [-1,1], P << 1 so S ≈ P (95.3% of values < 0.01);
the feature is effectively linear mel power and strongly depends on per-speaker recording gain
(mean S: jackson 0.0150 … yweweler 0.0003). Full spec run (60 epochs, src/train_classifier.py):

| fold (test speaker) | 1 george | 2 jackson | 3 lucas | 4 nicolas | 5 theo | 6 yweweler | mean |
|---|---|---|---|---|---|---|---|
| spec log1p(P), 60 ep (official run) | 0.182 | 0.276 | 0.230 | 0.354 | 0.106 | 0.160 | 0.218 |

Quick diagnostics (diag.py 30 epochs / diag2.py 40 epochs, same classifier, Val-selected checkpoint):

| variant | f1 | f2 | f3 | f4 | f5 | f6 | mean |
|---|---|---|---|---|---|---|---|
| spec_log1p (S = log1p(P)) | 0.200 | 0.278 | 0.280 | 0.368 | 0.116 | 0.102 | 0.224 |
| log_eps (S = log(P + 1e-6)) | 0.466 | 0.462 | 0.276 | 0.362 | 0.184 | 0.324 | 0.346 |
| peak_log1p (audio peak-normalised, then log1p) | 0.322 | 0.670 | 0.300 | 0.462 | 0.420 | 0.468 | 0.440 |
| peak_log_eps (peak-normalised, log(P+1e-6)) | 0.430 | 0.396 | 0.362 | 0.370 | 0.516 | 0.402 | 0.413 |
| log_eps + per-recording CMVN | 0.498 | 0.652 | 0.212 | 0.322 | 0.408 | 0.496 | 0.431 |
| peak_log1p + time-shift/noise augmentation | 0.290 | 0.682 | 0.444 | 0.404 | 0.384 | 0.436 | 0.440 |
| spec_log1p + CMVN + augmentation | 0.438 | 0.472 | 0.240 | 0.478 | 0.318 | 0.478 | 0.404 |
| log_eps + CMVN + augmentation | 0.470 | 0.552 | 0.308 | 0.276 | 0.322 | 0.492 | 0.403 |

Chance level = 0.10. These runs were NOT used to select anything on Test; they only document the problem.

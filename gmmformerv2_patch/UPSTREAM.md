# GMMFormerV2 source changes

The upstream repository is [GMMFormerV2](https://github.com/huangmozhi9527/GMMFormer_v2), revision 5bacac96f77408c2bef27c25637e2cf67fb062b9. The upstream repository specifies no license at this revision. This directory distributes the changes in patch form.

Six removed configuration lines contain machine-specific paths in upstream. The patch identifies those lines by path and line number in the pinned upstream revision. The helper reads them from your upstream clone and emits the complete Git patch. From the root of that clone, run the following in a shell with byte-preserving pipes.

```sh
python /path/to/gmmformerv2_patch/materialize.py . | git apply --check -
python /path/to/gmmformerv2_patch/materialize.py . | git apply -
```

The resulting source tree matches the source used for the experiments. The retraining variants also use experiment-specific code under `scripts/`.

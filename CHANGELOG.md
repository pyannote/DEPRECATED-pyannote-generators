### Version 0.12.1 (2017-05-17)

  - fix: fix corner case in "random_label_index"

### Version 0.12 (2017-05-11)

  - chore: reduce BaseBatchGenerator set of pack_xxxx methods
  - feat: add "repeat" option to "random_label_index"
  - setup: switch to pyanote.core 0.13.2

### Version 0.11.1 (2017-04-21)

  - fix: fix batch packing when signature is a dictionary

### Version 0.11 (2017-04-20)

  - feat: add "batchify" utility function

### Version 0.10 (2017-02-07)

  - feat: add "skip_unlabeled" option to labeled segment generator
  - setup: switch to pyannote.core 0.13

### Version 0.9.2 (2017-01-25)

  - feat: add "incomplete" option to yield incomplete final batch
  - feat: add support for heterogeneous segments
  - setup: switch to pyannote.core 0.11
  - BREAKING: rename "coverage" source to "support"

### Version 0.7 (2017-01-11)

  - feat: add support for approximate 'annotated' source
  - setup: switch to pyannote.database 0.11

### Version 0.6.2 (2016-12-16)

  - feat: default 'step' to half of 'duration'

### Version 0.6 (2016-12-14)

  - setup: switch to pyannote.database 0.9
  - improve: use new 'get_unique_identifier' function

### Version 0.5 (2016-12-08)

  - feat: add option to 'robustify' FileBasedBatchGenerator
  - setup: switch to pyannote.database 0.8

### Version 0.4.2 (2016-12-05)

  - setup: switch to pyannote.database 0.7.1
  - feat: skip files for which preprocessing fails

### Version 0.4.1 (2016-11-24)

  - feat: add min_duration support to sliding segments generator

### Version 0.3 (2016-11-16)

  - feat: add random_label_index generator

### Version 0.2 (2016-11-14)

  - feat: add variable-duration segment generator
  - setup: switch to pyannote.core 0.8

### Version 0.1.1 (2016-09-22)

  - fix: be consistent with pyannote.database generators

### Version 0.1 (2016-09-21)

  - first public version

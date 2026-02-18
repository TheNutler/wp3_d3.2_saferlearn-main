# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Actions

- Added
- Changed
- Deprecated
- Removed
- Fixed
- Security

## 28/09/2022

### Added

- Version 2.2 of pate_he
- Parameters (nb_classes, nb_teachers) set through the ui instead of hardcoding

### Changed

- ui
  - refactor code
  - build the server in the image build and serve it in a lightweight image
- call pate_he binaries in Python instead of depending on the shell scripts
- clean the repository

### Fixed

- Correct the 0 value of the alpha security parameter

### Security

- Updated all ui dependencies

## 02/11/2022

### Added

- Development container support

### Changed

- keys deployment
  - HE keys are now communicated with Kafka instead of using the commonly share filesystem

### Fixed

- Python code quality
  - cleaning and formatting
  - linting
  - typing

## 01/12/2022

### Changed

- Moved from `os` filesystem management to `pathlib`

### Fixed

- Match order of decryption with order of aggregation

## 27/04/2023 v1.2.0

### Added

- clear PATE
- Differential Privacy for clear and HE PATE
  - pate_hev2.3 integration
  - added DP parameters selection in UI
- computation duration displayed
- UC2 and UC3
- FL support
- air-gapped build support

### Changed

- saferlearn.sh bash script to control the project
- HE aggregation is now performed by batch to avoid file handles leakage
- usecases are now implemented in separate files

### Deprecated

- all bash scripts other than saferlearn.sh are deprecated and to be deleted in the future

### Removed

- remove unused file data_import_phase
- remove unused function in utilities.py

### Fixed

- orchestrator now wait for the correct number of messages

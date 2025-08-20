# Contributing to OVOD

Thanks for your interest in contributing to the Open-Vocabulary Object Detection (OVOD) project! 

## Development Setup

1. **Environment Setup**
   ```bash
   conda env create -f env-ovod.yml
   conda activate ovod
   ```

2. **Data Setup**
   ```bash
   make link-data  # Creates symlink to COCO dataset
   ```

3. **Verify Installation**
   ```bash
   make test       # Run basic tests
   make eval-50    # Quick evaluation
   ```

## Development Workflow

### Code Style
- We use `black` for code formatting
- Follow PEP 8 conventions
- Add type hints where appropriate
- Keep functions focused and well-documented

### Testing
- Add tests for new functionality in `tests/`
- Run the test suite: `pytest tests/`
- Ensure CI passes before submitting PRs

### Evaluation
- Test changes with multiple prompt strategies:
  ```bash
  make eval-person    # Person detection
  make eval-common    # General objects  
  make eval-full-prompt  # Full COCO evaluation
  ```

### Performance Benchmarking
- Profile latency on your hardware
- Document performance changes in PR descriptions
- Aim to maintain <500ms/image on modern GPUs

## Submitting Changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with appropriate tests
4. Ensure all tests pass: `pytest tests/`
5. Update documentation if needed
6. Submit a pull request with a clear description

### PR Requirements
- [ ] Tests pass locally and in CI
- [ ] Code follows project style guidelines
- [ ] Performance impact documented (if applicable)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (for significant changes)

## Types of Contributions

### Bug Fixes
- Include reproduction steps in the issue
- Add regression tests where possible
- Keep changes minimal and focused

### New Features
- Open an issue first to discuss the approach
- Consider impact on existing prompt strategies
- Ensure backward compatibility when possible

### Performance Improvements
- Benchmark before and after changes
- Test on different hardware if possible
- Document any trade-offs

### Documentation
- Fix typos, improve clarity
- Add examples for complex features
- Update setup instructions as needed

## Code Areas

### Core Pipeline (`ovod/pipeline.py`)
- Main orchestration logic
- Model loading and state management
- Error handling and recovery

### Detection (`src/detector.py`)
- Grounding DINO integration
- Text prompt processing
- Box format handling

### Segmentation (`src/segmenter.py`)
- SAM 2 integration
- Mask generation and processing

### Evaluation (`eval.py`)
- COCO evaluation metrics
- Box format auto-detection
- Prompt strategy testing

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed information for better assistance

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
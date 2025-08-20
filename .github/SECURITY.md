# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in OVOD, please report it responsibly:

1. **Do not** open a public issue for security vulnerabilities
2. **Email** security reports to: robertlupo1997@gmail.com
3. **Include** detailed information about the vulnerability
4. **Provide** steps to reproduce if possible

### What to expect

- **Response time**: Within 48 hours
- **Assessment**: We will evaluate the impact and severity
- **Resolution**: Security patches will be prioritized
- **Acknowledgment**: Contributors will be credited (if desired)

### Security considerations

- Model weights and checkpoints are downloaded from external sources
- User inputs (text prompts) are processed by AI models
- Image uploads in demo mode should be treated with appropriate caution
- Dependencies (GroundingDINO, SAM2) inherit their own security considerations

Thank you for helping keep OVOD secure!
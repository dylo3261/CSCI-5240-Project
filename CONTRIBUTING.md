# Contributing Guidelines

    ## Branching Strategy
    - `main` - Production code
    - `develop` - Development branch
    - `feature/description` - New features
    - `bugfix/description` - Bug fixes

    ## Workflow
    1. Create a feature branch from `develop`
    2. Make your changes
    3. Write tests
    4. Create a Pull Request to `develop`
    5. Get 1 approval
    6. Merge after CI passes

    ## Commit Message Format
```
    type(scope): subject

    Examples:
    feat(lambda): add location caching
    fix(model): correct prediction threshold
    docs(readme): update setup instructions
```

    ## Code Standards
    - Run `black` for formatting
    - Run `flake8` for linting
    - Write unit tests for new code
    - Update documentation
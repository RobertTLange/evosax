# Contributing to `evosax`

Thank you for your interest in contributing to `evosax`! We deeply appreciate you taking the time to help make `evosax` better. Whether you're contributing code, suggesting new features, opening an issue, improving documentation or writing tutorials - all contributions are valuable and welcome.

We also appreciate if you spread the word, for instance by starring the `evosax` GitHub repository, or referencing `evosax` in projects that used it.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the `evosax` repository by clicking the Fork button on the repository page. This creates a copy of the `evosax` repository in your own account.

2. Install Python >= 3.10 locally in order to run tests.

3. `pip` installing your fork from source. This allows you to modify the code and immediately test it out:

```bash
git clone https://github.com/RobertTLange/evosax
cd evosax
pip install -e ".[dev]"  # Installs evosax from the current directory in editable mode.
```

4. Add the `evosax` repository as an upstream remote, so you can use it to sync your changes.

```bash
git remote add upstream https://github.com/RobertTLange/evosax
```

5. Create a branch where you will develop from:

```bash
git checkout -b name-of-change
```

And implement your changes using your favorite editor.

6. Make sure your code passes `evosax` lint and type checks, by running the following from the top of the repository:

```bash
ruff format .
ruff check .
```

7. Make sure the tests pass by running the following command from the top of the repository:

```bash
pytest tests/
```

8. Once you are satisfied with your change, create a commit as follows ( how to write a commit message):

```bash
git add file1.py file2.py ...
git commit -m "Your commit message"
```

Then sync your code with the main repo:

```bash
git fetch upstream
git rebase upstream/main
```

Finally, push your commit on your development branch and create a remote branch in your fork that you can use to create a pull request from:

```bash
git push --set-upstream origin name-of-change
```

9. Create a pull request from the `evosax` repository and send it for review.

## Report a bug or suggest a new feature using GitHub issues

Go to https://github.com/RobertTLange/evosax/issues and click on "New issue".

Informative bug reports tend to have:

- A quick summary
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Additional notes

## License

By contributing, you agree that your contributions will be licensed under its Apache license.

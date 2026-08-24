# Security

Do not report EPİAŞ usernames, passwords, TGT values or other credentials in public issues.

The repository intentionally reads credentials only from environment variables. If a credential is accidentally committed, revoke/rotate it immediately and remove it from Git history; deleting the visible file alone is not sufficient.

For code-level security problems, use a private GitHub security advisory when available rather than a public issue.

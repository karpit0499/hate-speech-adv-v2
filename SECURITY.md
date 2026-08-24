# Security policy

Please report vulnerabilities privately through this repository's **Security** tab by
opening a private vulnerability report. Do not include exploit details in a public issue.

The browser demo performs inference locally and does not require an API key. The serving
API stores raw input text only when an operator explicitly opts in with `STORE_RAW_TEXT=true`;
the default is redaction.

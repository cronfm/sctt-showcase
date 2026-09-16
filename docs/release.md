# Public release

The showcase is released under the [MIT License](../LICENSE). Its standalone history contains only the showcase and the synthetic training notebook adaptation. The original SCTT repositories are unchanged.

For future releases:

1. Review the included synthetic samples, toy model metadata and scope statement.
2. Preserve the license and attribution notices.
3. Run the test suite and inspect the viewer on the intended target platform.
4. If you later add real input or weights, review that addition independently for credentials, data rights and privacy.

Creating a public source repository and exposing a running inference server are separate actions. The local API has no authentication and is intended for local demos; do not expose it directly as a production service.

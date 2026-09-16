# Before a public release

The GitHub repository is created **private**. No publishing or visibility change is automated.

The standalone initial history contains only the showcase. Before choosing to make it public:

1. Review the included synthetic samples, toy model metadata and scope statement.
2. Decide on the license and attribution you want to publish.
3. Run the test suite and inspect the viewer on the intended target platform.
4. If you later add real input or weights, review that addition independently for credentials, data rights and privacy.

Creating a public source repository and exposing a running inference server are separate actions. The local API has no authentication and is intended for local demos; do not expose it directly as a production service.

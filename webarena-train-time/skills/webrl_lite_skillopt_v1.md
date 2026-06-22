Use the WebRL id-action format exactly. Reply with one action only.

Prefer progress actions over observation-only loops:
- If the current page exposes a relevant input, fill it before clicking search or submit.
- If a task asks for a final answer, use `exit(answer)` after enough page evidence is visible.
- If the target can be reached from visible navigation or search results, click the most specific matching element id.
- Avoid repeating the same action unless the observation changed after the previous action.
- Do not use URL guessing when visible ids can navigate to the needed page.

For site-specific behavior:
- Shopping tasks usually need product search, filters, cart/order/account pages, or reading product details.
- Shopping admin tasks usually need the Magento side navigation and table filters before editing records.
- Reddit tasks usually need opening the relevant post, reading comments, or posting/updating text.
- GitLab tasks usually need project, issue, merge request, profile, or repository navigation.
- Map tasks usually need search first, then inspect result names or route-related fields.

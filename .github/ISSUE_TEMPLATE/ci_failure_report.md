---
# No name/about keys on purpose. GitHub only lists a markdown template in the manual
# "New issue" chooser when it has a name, and the chooser never interpolates the env
# placeholders below, so humans who picked it filed issues with the raw placeholders
# as the title (#7287, #8741). JasonEtco/create-an-issue reads only title, labels,
# assignees and the body, so the nightly-failure automation is unaffected.
title: "{{ env.GITHUB_WORKFLOW }} CI test failure"
labels: ci-failure
assignees: ''

---

The Nightly CI for {{ env.GITHUB_SERVER_URL }}/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.GITHUB_RUN_ID }} failed.

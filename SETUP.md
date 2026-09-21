# Setup checklist

## 1. Upload the prepared files

Copy `README.md`, `assets/dark.svg`, `assets/light.svg`, and `.github/workflows/snake.yml` into the `main` branch of `yousefJooX/yousefJooX`.

## 2. Enable the contribution snake

Open the profile repository (not account) settings:

`https://github.com/yousefJooX/yousefJooX/settings/actions`

Under **Workflow permissions**, select **Read and write permissions** and save. Then open **Actions**, select **Generate Snake Animation**, and run it once. Only expect the snake image to appear after that run is green and the `output` branch exists.

## 3. Self-host the stats cards

The README currently uses the public stats endpoint so it renders immediately. For reliable cards:

1. Create a GitHub classic token from Settings > Developer settings > Personal access tokens > Tokens (classic). Use the `repo` scope and copy it immediately.
2. Never paste the token into chat, a public repository, or the README.
3. Fork `anuraghazra/github-readme-stats`.
4. Import the fork into a free Vercel project.
5. Add the environment variable `PAT_1` in Vercel, with the token as its value, then deploy.
6. Replace both occurrences of `https://github-readme-stats.vercel.app` in `README.md` with the private Vercel deployment URL.

`hide_rank=true` is intentional: the rank strongly reflects stars and followers rather than coding skill.

## 4. Verify both themes

Switch GitHub between dark and light appearance and reload the profile. If an updated SVG looks stale, append `?v=999` to its raw URL and inspect the source; GitHub's CDN may cache the old file for a while.

## Regenerating the banner later

The source portrait, logo masks, generator, and `.npy` dot maps are included. From the repository root, install Pillow, NumPy, and SciPy, then run:

```bash
python3 tools/generate_profile_assets.py
```

The generator rewrites `assets/dark.svg`, `assets/light.svg`, and the two portrait map files.

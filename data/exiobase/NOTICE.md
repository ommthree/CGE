# EXIOBASE supply-share artifact — source & licence

`supply_shares_2019.json` in this directory is **derived data**: EXIOBASE product→industry
production shares computed from the EXIOBASE 3 MRSUT (monetary multi-regional supply-use table)
supply matrix. It is **not** produced by this project; it is a compact, reviewable transform of
EXIOBASE, redistributed here under the licence below.

## What it is

For each of EXIOBASE's 200 products, the fraction of that product's total monetary supply produced by
each of the 163 industries, summed over all regions and normalised to sum to 1 (shares below 1e-4
dropped and the remainder renormalised). The nature product-bridge uses these **observed** shares to
map a product build's ENCORE dependencies through the industry-keyed crosswalk, instead of inferring
the producing industry from a classification code prefix. 16 products (recycling/treatment residuals,
extra-territorial bodies) have no market supply in the SUT and carry an empty share; the bridge falls
back to the code-prefix method for those (recorded in the `ProductBridgeAudit`).

## Attribution (required by the licence)

> **EXIOBASE 3** — Stadler, K., Wood, R., Bulavskaya, T., et al. (2018). *EXIOBASE 3: Developing a
> Time Series of Detailed Environmentally Extended Multi-Regional Input-Output Tables.* Journal of
> Industrial Ecology 22(3), 502–515.

- **Version:** EXIOBASE 3.8.2, MRSUT vers 20210125, year 2019 (current-price Euro).
- **DOI / source:** <https://doi.org/10.5281/zenodo.5589597> (open-access Zenodo record).
- **Retrieved:** 2026-08-14. **Regenerate with:** `python scripts/build_supply_shares.py`.

## Licence — CC BY-SA 4.0

EXIOBASE is released under a **Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)** licence. This derived artifact is redistributed under the same terms, with
attribution as above. Full licence text: <https://creativecommons.org/licenses/by-sa/4.0/>

**ShareAlike caveat.** As with the ENCORE data (see `data/encore/NOTICE.md`), CC BY-SA 4.0 is
copyleft: a work that *adapts* this data may itself have to be offered under CC BY-SA 4.0. Fine for
this research prototype; confirm with legal counsel before any commercial/closed distribution that
embeds or adapts it. This NOTICE is a developer's summary, not legal advice.

## What is vendored (and what is not)

- **Vendored:** the small derived `supply_shares_2019.json` (~0.6 MB) and its provenance.
- **NOT vendored:** the raw multi-GB MRSUT archive (`downloads/exiobase/MRSUT_2019.zip`, gitignored)
  and the raw IOT. Only the derived shares are committed.

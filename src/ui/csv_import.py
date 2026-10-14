"""Two-step CSV import UI: upload, review the mapping and the rejects, commit.

The step that matters is the second one. :mod:`src.portfolio_importer` can map
"Tradingsymbol" onto ``ticker`` and "Average Price" onto ``buy_price``, but a
mapping is an inference, and an inference about which column holds your cost
basis must be shown to the user before it becomes a portfolio. So the preview
is not decoration — nothing is written to session state until **Import** is
pressed.

The panel always shows these, whether or not they contain bad news:

* the column mapping, in the user's own column names, labelled by how
  confident the importer is about each field
* a per-field picker to correct any of them, because detection cannot cover
  every header that exists and a mis-read column must not be a dead end
* every row that will be imported
* every row that will NOT be, with the reason

Used at two sizes: ``compact=True`` in the sidebar rail (verdict only — a
two-column picker is unusable at 330px), full width on the Portfolio page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import isin as isin_mod
from src import portfolio_importer as importer
from src.ui import components as ui
from src.ui import data_access
from src.ui import session as sess

__all__ = ["import_panel"]

#: Session key prefix holding a staged (uploaded, parsed, not yet committed)
#: preview. Keyed per panel so the sidebar and the page never collide.
_STAGE = "import_stage_"

#: Label for "this field is not in the file" in the mapping editor. A visible
#: option rather than a blank one, so unmapping a field is a deliberate choice.
_NOT_PRESENT = "— not present —"


def _stage_preview(stage_key: str, preview: importer.ImportPreview, bump: bool = False) -> None:
    """Replace the staged preview in place, keeping the file identity.

    ``bump`` advances ``map_version``, which is what forces the mapping
    editor's widgets to re-read their defaults from the new mapping instead of
    showing Streamlit's remembered selection.
    """
    staged = dict(st.session_state.get(stage_key) or {})
    staged["preview"] = preview
    if bump:
        staged["map_version"] = int(staged.get("map_version", 0)) + 1
    st.session_state[stage_key] = staged


def _tone_for(preview: importer.ImportPreview) -> str:
    if preview.fatal:
        return "loss"
    if preview.n_rejected or preview.needs_review:
        return "warn"
    return "gain"


def _render_summary(preview: importer.ImportPreview) -> None:
    """The ✓ / ⚠ / ✗ checklist, then the KPI row."""
    for line in preview.summary_lines():
        if line.startswith("✗"):
            st.error(line)
        elif line.startswith("⚠"):
            st.warning(line)
        else:
            st.success(line)

    if preview.fatal:
        return

    ui.kpi_row(
        [
            {
                "label": "Rows in file",
                "value": f"{preview.source_rows:,}",
                "tone": "neutral",
                "footnote": "excluding the header",
            },
            {
                "label": "Will import",
                "value": f"{preview.n_rows:,}",
                "tone": _tone_for(preview),
                "footnote": f"{preview.n_tickers} distinct symbols",
            },
            {
                "label": "Cannot import",
                "value": f"{preview.n_rejected:,}",
                "tone": "loss" if preview.n_rejected else "neutral",
                "footnote": "listed with reasons below",
            },
        ],
        columns=3,
    )


def _mapping_editor(preview: importer.ImportPreview, key: str, stage_key: str) -> None:
    """Per-field column picker, so a mis-detected header is fixable in place.

    Detection covers the brokers we know about plus a similarity fallback, but
    a header nobody anticipated ("Cost of acquisition per unit") still lands on
    ``✗ Could not find a column for: buy_price`` — which without this was a
    dead end that sent the user back to their spreadsheet to rename a column.

    Wrapped in a form on purpose: without one, every selectbox change would
    rerun the script and re-derive the whole preview, so a three-field
    correction would recompute twice for the two intermediate states nobody
    asked about. One Apply, one rebuild.
    """
    # Nothing to re-point when the file never parsed into columns (an
    # unreadable upload, a wrong file type). Offering an empty picker there
    # would imply the problem is a mapping mistake when it is not.
    if not preview.source_columns or preview.source_frame is None:
        return

    schema, required = (
        (importer.TRANSACTION_COLUMNS, importer.TRANSACTION_REQUIRED)
        if preview.kind == "transactions"
        else (importer.HOLDING_COLUMNS, importer.HOLDING_REQUIRED)
    )
    options = [_NOT_PRESENT, *preview.source_columns]
    # The form key is versioned so the selectboxes re-read their defaults from
    # the NEW mapping after an Apply. Reusing the key would leave Streamlit's
    # widget state showing the value the user picked even where the override
    # was dropped (a column that is not in the file), which reads as a bug.
    version = int(st.session_state.get(stage_key, {}).get("map_version", 0))

    with st.form(key=f"{key}_map_form_{version}"):
        st.caption(
            "Point a field at a different column when the automatic guess is wrong. "
            "Fields marked \\* are required. One column can only feed one field — "
            "reassigning it takes it away from whatever held it."
        )
        picked: dict[str, str] = {}
        columns = st.columns(2)
        for index, canonical in enumerate(schema):
            current = preview.mapping.source_of(canonical) or _NOT_PRESENT
            picked[canonical] = columns[index % 2].selectbox(
                f"{canonical}{' *' if canonical in required else ''}",
                options,
                index=options.index(current) if current in options else 0,
                key=f"{key}_map_{canonical}_{version}",
                help=f"Detected as: {current}",
            )
        left, right = st.columns([2, 1])
        applied = left.form_submit_button("Apply mapping", width="stretch")
        reset = right.form_submit_button(
            "Reset to auto", width="stretch", help="Discard every manual override."
        )

    if reset:
        _stage_preview(stage_key, importer.redetect(preview), bump=True)
        st.rerun()

    if not applied:
        return

    # Send only what actually changed, so `mapping.overridden` stays a record
    # of the user's real decisions rather than of every field on the form.
    overrides: dict[str, str | None] = {
        canonical: (None if chosen == _NOT_PRESENT else chosen)
        for canonical, chosen in picked.items()
        if (preview.mapping.source_of(canonical) or _NOT_PRESENT) != chosen
    }
    if not overrides:
        st.info("Nothing changed — the mapping already matches your selection.")
        return

    _stage_preview(
        stage_key,
        importer.remap(preview, overrides, base_currency=sess.get_config().base_currency),
        bump=True,
    )
    st.rerun()


def _isin_section(preview: importer.ImportPreview, key: str, stage_key: str) -> None:
    """Report ISIN resolution, and offer a provider lookup for what failed.

    An unresolved ISIN is a *different* problem from a bad number, and it has a
    different fix — look the identifier up, rather than correct a value — so it
    gets its own section rather than being buried in the reject table.
    """
    if preview.isin_resolved:
        with st.expander(
            f"ISINs resolved to symbols ({len(preview.isin_resolved)})", expanded=False
        ):
            summary = isin_mod.map_summary()
            st.caption(
                "Resolved from the local ISIN table — "
                f"{summary.get('seed', 0)} bundled and {summary.get('learned', 0)} learned "
                "from your own earlier imports. Nothing here is guessed: an ISIN either "
                "appears in that table or it is reported as unresolved."
            )
            ui.dataframe(
                pd.DataFrame(sorted(preview.isin_resolved.items()), columns=["ISIN", "Symbol"])
            )

    if not preview.isin_unresolved:
        return

    with st.expander(
        f"ISINs that could not be mapped ({len(preview.isin_unresolved)})", expanded=True
    ):
        st.caption(
            "These are valid ISINs — the check digit passes — but the local table has "
            "no symbol for them, so their rows cannot be imported. The bundled table is "
            "deliberately small; the two ways to fix this are to include a symbol column "
            "in your export, or to look them up online."
        )
        for code in preview.isin_unresolved:
            country = isin_mod.country_name(code)
            st.markdown(f"- `{code}`" + (f" — issued in {country}" if country else ""))

        if st.button(
            f"Look up {len(preview.isin_unresolved)} ISIN(s) online",
            key=f"{key}_isin_lookup",
            help="Query Yahoo Finance for these identifiers. Needs an internet "
            "connection. Anything found is remembered for future imports.",
        ):
            with st.spinner("Looking up ISINs…"):
                found, errors = data_access.lookup_isins_online(preview.isin_unresolved)

            if found:
                stored, problems = isin_mod.learn(found, source="provider")
                st.success(
                    f"Resolved and remembered {stored} ISIN(s): "
                    + ", ".join(f"`{k}` → `{v}`" for k, v in found.items())
                )
                if problems:
                    ui.warning_list(problems, "Mappings that could not be stored")
                # Re-derive from scratch so the newly-learned mappings apply to
                # every row, and so the reject list shrinks accordingly.
                _stage_preview(
                    stage_key,
                    importer.redetect(preview, base_currency=sess.get_config().base_currency),
                    bump=True,
                )
                st.rerun()

            if errors:
                ui.warning_list(
                    [f"{code}: {reason}" for code, reason in errors.items()],
                    "ISINs the provider could not resolve",
                )
            if not found and not errors:
                st.warning("The lookup returned nothing. Check your internet connection.")


def _raw_head(preview: importer.ImportPreview, rows: int = 5) -> None:
    """The first few rows of the file exactly as it was read.

    Shown only when nothing survived, which is precisely when the user is being
    asked to pick a column and has nothing to pick it *by*. Choosing between
    "Rate" and "Cost of acquisition per unit" from the header names alone is
    guesswork; seeing that one column holds 1182.95 and the other holds 90
    settles it immediately.
    """
    frame = preview.source_frame
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    if not preview.frame.empty:
        return
    st.caption(f"Your file as it was read — first {min(rows, len(frame))} row(s):")
    ui.dataframe(frame.head(rows))


def _render_detail(preview: importer.ImportPreview, key: str, stage_key: str) -> None:
    """Mapping table + editor, the rows to import, the rejects, the review list."""
    mapping = preview.mapping
    with st.expander(
        "Column mapping",
        expanded=bool(mapping.fuzzy or mapping.blocked() or preview.fatal),
    ):
        st.caption(
            "How each of your columns was read. *needs review* means it was matched by "
            "similarity rather than by a name the importer knows; *you chose this* means "
            "you set it by hand below."
        )
        ui.dataframe(mapping.as_frame())
        _raw_head(preview)
        _mapping_editor(preview, key, stage_key)

    if not preview.frame.empty:
        st.caption(f"Preview — the {preview.n_rows} row(s) that will be imported:")
        ui.dataframe(preview.frame, height=min(38 * (preview.n_rows + 1) + 8, 340))

    if not preview.rejected.empty:
        with st.expander(
            f"Rows that cannot be imported ({preview.n_rejected}) — nothing is dropped silently",
            expanded=True,
        ):
            display = preview.rejected.rename(columns={"_row": "File row", "_reason": "Why"})
            front = [c for c in ("File row", "Why") if c in display.columns]
            ui.dataframe(display[front + [c for c in display.columns if c not in front]])
            st.download_button(
                "Download rejected rows as CSV",
                data=preview.rejected.to_csv(index=False).encode("utf-8"),
                file_name=f"rejected_{preview.kind}.csv",
                mime="text/csv",
                key=f"{key}_reject_dl",
                help="Fix them in your spreadsheet and re-upload just those rows.",
            )

    if preview.needs_review:
        with st.expander(f"Symbols to confirm ({len(preview.needs_review)})", expanded=False):
            st.caption(
                "These are **not** errors. The bundled stock-master files cover common "
                "Indian and US listings; a symbol absent from them may still price "
                "perfectly well — only fetching it can settle that. Listed here so a "
                "typo is caught before you analyse."
            )
            for symbol in preview.needs_review:
                venues = preview.ambiguous.get(symbol)
                if venues:
                    st.markdown(
                        f"- `{symbol}` lists on several venues ({', '.join(f'`{v}`' for v in venues)}) "
                        f"— imported as `{preview.resolved.get(symbol, symbol)}`"
                    )
                else:
                    st.markdown(f"- `{symbol}` not found in the bundled files")

    _isin_section(preview, key, stage_key)

    warnings = preview.warnings()
    if warnings:
        ui.warning_list([str(w) for w in warnings], "Values that were adjusted")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def import_panel(
    kind: importer.ImportKind,
    key: str,
    label: str | None = None,
    compact: bool = False,
) -> None:
    """Upload → preview → commit, for holdings or a trade ledger.

    ``key`` namespaces every widget and the staged preview, so several panels
    can live on one screen. Nothing reaches ``st.session_state``'s holdings or
    transactions until the user presses Import.
    """
    is_txn = kind == "transactions"
    heading = label or ("Trade ledger" if is_txn else "Holdings")
    stage_key = f"{_STAGE}{key}"

    uploaded = st.file_uploader(
        heading,
        type=["csv", "xlsx", "xls"],
        key=f"{key}_file",
        help=(
            "Any broker export. Column names are matched automatically — "
            + (
                "`trade_date`/`transaction_date`/`Date`, `action`/`Side`/`Trade Type`, "
                "`qty`/`Shares`, `price`/`Execution Price`, `fees`/`Brokerage`."
                if is_txn
                else "`ticker`/`Symbol`/`Tradingsymbol`, `qty`/`Quantity`/`Shares`, "
                "`buy_price`/`Average Price`/`Avg. Cost`."
            )
        ),
    )

    # Parse once per uploaded file. st.file_uploader hands back the same object
    # on every rerun, so without the file-id guard the user's review clicks
    # would re-parse the file (and re-scroll the panel) on every interaction.
    if uploaded is not None:
        staged = st.session_state.get(stage_key)
        if staged is None or staged.get("file_id") != uploaded.file_id:
            read = importer.import_transactions if is_txn else importer.import_holdings
            with st.spinner(f"Reading {uploaded.name}…"):
                parsed = read(
                    uploaded,
                    uploaded.name,
                    base_currency=sess.get_config().base_currency,
                )
            st.session_state[stage_key] = {
                "file_id": uploaded.file_id,
                "name": uploaded.name,
                "preview": parsed,
                # Reset the mapping-editor widget generation for a new file:
                # the previous file's column names are not this one's options.
                "map_version": 0,
            }

    staged = st.session_state.get(stage_key)
    if not staged:
        if is_txn:
            st.caption(
                "Optional. A ledger unlocks FIFO tax lots, realised P&L and a "
                "money-weighted return (XIRR)."
            )
        return

    preview: importer.ImportPreview = staged["preview"]
    st.markdown(f"**{staged['name']}** — {preview.source_rows:,} data row(s)")

    if compact:
        # The sidebar is ~330px wide: a mapping table and a preview grid are
        # unreadable in it. Show the verdict and send the user to the page
        # that has the room to actually review before committing.
        for line in preview.summary_lines():
            st.caption(line)
        if not preview.accounted_for:  # pragma: no cover - defensive
            st.error("Row accounting failed — review this import on the Portfolio page.")
        if preview.mapping.missing:
            st.caption(
                "A column could not be found. Fix the mapping under **Import from CSV** "
                "on the Portfolio › Build page — the sidebar is too narrow for the picker."
            )
    else:
        _render_summary(preview)
        _render_detail(preview, key, stage_key)

    if not preview.ok:
        if st.button("Discard", key=f"{key}_discard", width="stretch"):
            st.session_state.pop(stage_key, None)
            st.rerun()
        return

    noun = "transaction" if is_txn else "holding"
    left, right = st.columns([2, 1]) if not compact else (st, st)
    if left.button(
        f"Import {preview.n_rows} {noun}{'s' if preview.n_rows != 1 else ''}",
        type="primary",
        key=f"{key}_commit",
        width="stretch",
        help="Replaces the current table. Nothing is written until you press this.",
    ):
        if is_txn:
            sess.replace_transactions(preview.to_transactions())
        else:
            sess.replace_holdings(preview.to_holdings())
        st.session_state.pop(stage_key, None)
        st.toast(f"Imported {preview.n_rows} {noun}(s) from {staged['name']}.", icon="📥")
        st.rerun()

    if right.button("Cancel", key=f"{key}_cancel", width="stretch"):
        st.session_state.pop(stage_key, None)
        st.rerun()

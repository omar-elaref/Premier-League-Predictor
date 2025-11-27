import torch
from torch.utils.data import Subset


@torch.no_grad()
def print_validation_table_predicted(dataset, val_loader, model, device):
    
    model.eval()

    subset = val_loader.dataset  # this should be a Subset over the original dataset
    if isinstance(subset, Subset):
        val_indices = list(subset.indices)
    else:
        val_indices = list(range(len(dataset)))

    table = {} 

    def ensure_team(team):
        if team not in table:
            table[team] = {
                "P": 0,
                "W": 0,
                "D": 0,
                "L": 0,
                "GF": 0,
                "GA": 0,
            }

    offset = 0
    for batch in val_loader:
        # indices for this batch in the original dataset
        batch_size = batch["y"].size(0)
        batch_indices = val_indices[offset: offset + batch_size]
        offset += batch_size

        # move batch to device
        for k in list(batch.keys()):
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)

        # predictions
        preds = model(
            batch["team1_ids"],
            batch["team2_ids"],
            batch["ground_flags"],
            batch["meta_numeric"],
            batch["h2h_seq"],
            batch["team1_seq"],
            batch["team2_seq"],
        )  # (B, 2)

        # round to nearest integer goals, clamp at 0
        pred_goals = torch.round(preds).clamp(min=0).long().cpu().numpy()

        for idx_in_batch, data_idx in enumerate(batch_indices):
            match = dataset.matches_df.iloc[data_idx]
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            hg = int(pred_goals[idx_in_batch, 0])
            ag = int(pred_goals[idx_in_batch, 1])

            ensure_team(home)
            ensure_team(away)

            # update games played
            table[home]["P"] += 1
            table[away]["P"] += 1

            # goals for/against
            table[home]["GF"] += hg
            table[home]["GA"] += ag
            table[away]["GF"] += ag
            table[away]["GA"] += hg

            # result from predictions
            if hg > ag:
                table[home]["W"] += 1
                table[away]["L"] += 1
            elif hg < ag:
                table[home]["L"] += 1
                table[away]["W"] += 1
            else:
                table[home]["D"] += 1
                table[away]["D"] += 1

    # convert to list and add GD / Points
    rows = []
    for team, s in table.items():
        GD = s["GF"] - s["GA"]
        Pts = 3 * s["W"] + 1 * s["D"]
        rows.append({
            "Team": team,
            "P": s["P"],
            "W": s["W"],
            "D": s["D"],
            "L": s["L"],
            "GF": s["GF"],
            "GA": s["GA"],
            "GD": GD,
            "Pts": Pts,
        })

    # sort by Points desc, GD desc, GF desc
    rows.sort(key=lambda r: (r["Pts"], r["GD"], r["GF"]), reverse=True)

    # print nicely
    print("\n=== Validation League Table (Predicted Results) ===")
    header = f"{'Pos':>3}  {'Team':<20} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'GF':>3} {'GA':>3} {'GD':>3} {'Pts':>3}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:>3}  {r['Team']:<20} "
            f"{r['P']:>2} {r['W']:>2} {r['D']:>2} {r['L']:>2} "
            f"{r['GF']:>3} {r['GA']:>3} {r['GD']:>3} {r['Pts']:>3}"
        )

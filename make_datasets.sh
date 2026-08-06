#!/usr/bin/env bash
#
# Build grasp datasets end to end: generate -> RRT expand -> collision filter -> plot.
#
#   ./make_datasets.sh                          # every default object, every stage
#   ./make_datasets.sh cube_40mm knife          # just these two
#   ./make_datasets.sh -v v2 -c 3.0             # version v2, tighter canonical space
#   ./make_datasets.sh --stages filter,plot     # re-filter datasets that already exist
#   ./make_datasets.sh --dry-run                # print the commands, run nothing
#
# Object names are the mesh basenames in my_assets/objects/ with the `_m.stl` dropped,
# so `knife` means `my_assets/objects/knife_m.stl`.
#
# Stages are skipped when their output directory already exists; pass --force to redo
# them. That makes the script safe to re-run after a crash, and cheap to extend with a
# new object without touching the ones already built.

set -euo pipefail

# ---------------------------------------------------------------- defaults
ROBOT=hsl_leap
VERSION=v1
CONCENTRATION=2.0          # 1.0 = uniform over the canonical box; 2-3 concentrates
N_GRASPS=100000            # generate stage
N_EXPANDED=500000          # RRT stage: TOTAL rows out, seed rows included
HUB_USER=iantc104          # empty (or --no-push) to keep everything local
STAGES=gen,rrt,filter,plot

OBJECT_PENETRATION=0.008   # [m] filter: reject deeper hand-inside-object than this
SELF_PENETRATION=0.005     # [m] filter: reject deeper hand-inside-hand than this
CENTER_SIGMA=0             # filter: >0 also thins toward the canonical centre

OBJECTS_DIR=my_assets/objects
DENSE_URDF=my_assets/hand/hsl_leap/urdf/leap_hand_right_dense_collision.urdf
OUT_ROOT=./outputs
LOG_DIR=./outputs/logs

DEFAULT_OBJECTS=(cube_40mm knife lightbulb rubber_duck wineglass_closed)

DRY_RUN=0
FORCE=0

usage() {
    # the leading comment block, minus the shebang
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    cat <<EOF

Options:
  -r, --robot NAME              robot in lygra/robot/__init__.py   [$ROBOT]
  -v, --version TAG             suffix on every output name        [$VERSION]
  -c, --concentration A         canonical-space concentration      [$CONCENTRATION]
  -n, --n-grasps N              rows out of the generate stage     [$N_GRASPS]
  -e, --n-expanded N            TOTAL rows out of the RRT stage    [$N_EXPANDED]
  -u, --hub-user NAME           push to NAME/<dataset>             [$HUB_USER]
      --no-push                 keep everything local
  -s, --stages A,B,C            any of gen,rrt,filter,plot         [$STAGES]
      --object-penetration M    filter threshold, metres           [$OBJECT_PENETRATION]
      --self-penetration M      filter threshold, metres           [$SELF_PENETRATION]
      --center-sigma S          filter: thin toward the centre     [$CENTER_SIGMA]
      --collision-urdf PATH     bone-bridged hand for the filter
      --out-root DIR            where datasets are written         [$OUT_ROOT]
  -f, --force                   redo stages whose output exists
      --dry-run                 print the commands, run nothing
  -h, --help                    this

Default objects: ${DEFAULT_OBJECTS[*]}
EOF
}

# ---------------------------------------------------------------- arguments
OBJECTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--robot)               ROBOT=$2; shift 2 ;;
        -v|--version)             VERSION=$2; shift 2 ;;
        -c|--concentration)       CONCENTRATION=$2; shift 2 ;;
        -n|--n-grasps)            N_GRASPS=$2; shift 2 ;;
        -e|--n-expanded)          N_EXPANDED=$2; shift 2 ;;
        -u|--hub-user)            HUB_USER=$2; shift 2 ;;
        --no-push)                HUB_USER=""; shift ;;
        -s|--stages)              STAGES=$2; shift 2 ;;
        --object-penetration)     OBJECT_PENETRATION=$2; shift 2 ;;
        --self-penetration)       SELF_PENETRATION=$2; shift 2 ;;
        --center-sigma)           CENTER_SIGMA=$2; shift 2 ;;
        --collision-urdf)         DENSE_URDF=$2; shift 2 ;;
        --out-root)               OUT_ROOT=$2; LOG_DIR=$2/logs; shift 2 ;;
        -f|--force)               FORCE=1; shift ;;
        --dry-run)                DRY_RUN=1; shift ;;
        -h|--help)                usage; exit 0 ;;
        -*)                       echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                        OBJECTS+=("$1"); shift ;;
    esac
done

[[ ${#OBJECTS[@]} -eq 0 ]] && OBJECTS=("${DEFAULT_OBJECTS[@]}")

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

for s in ${STAGES//,/ }; do
    case "$s" in
        gen|rrt|filter|plot) ;;
        *) echo "unknown stage: $s (want gen, rrt, filter or plot)" >&2; exit 2 ;;
    esac
done

# Fail on a bad object name now rather than an hour into the run.
for obj in "${OBJECTS[@]}"; do
    [[ -f "$OBJECTS_DIR/${obj}_m.stl" ]] || {
        echo "no mesh for '$obj' at $OBJECTS_DIR/${obj}_m.stl" >&2
        echo "available: $(ls "$OBJECTS_DIR"/*_m.stl 2>/dev/null | xargs -n1 basename \
             | sed 's/_m\.stl//' | tr '\n' ' ')" >&2
        exit 2
    }
done
if has_stage filter && [[ ! -f "$DENSE_URDF" ]]; then
    echo "collision URDF not found: $DENSE_URDF" >&2
    exit 2
fi

# ---------------------------------------------------------------- plumbing
run() {
    # run <log-name> <command...>
    local name=$1; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %s\n' "$*"
        return 0
    fi
    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/${name}.log"
    local start=$SECONDS
    if "$@" > >(tee "$log") 2>&1; then
        printf '  done in %dm%02ds  (log: %s)\n' $(((SECONDS-start)/60)) $(((SECONDS-start)%60)) "$log"
    else
        printf '  FAILED after %dm%02ds  (log: %s)\n' $(((SECONDS-start)/60)) $(((SECONDS-start)%60)) "$log" >&2
        return 1
    fi
}

# `--push_to_hub ""` is how every script here spells "do not push".
hub() { [[ -n $HUB_USER ]] && echo "$HUB_USER/$1" || echo ""; }

# Skip a stage whose output is already on disk, unless --force.
done_already() {
    [[ $FORCE -eq 0 && -d $1 ]] && { echo "  exists, skipping ($1) -- pass --force to redo"; return 0; }
    return 1
}

# ---------------------------------------------------------------- plan
cat <<EOF
robot            $ROBOT
version          $VERSION
stages           $STAGES
objects          ${OBJECTS[*]}
concentration    $CONCENTRATION $([[ $CONCENTRATION == 1.0 ]] && echo '(uniform over the whole canonical box)')
grasps           $N_GRASPS  ->  $N_EXPANDED after RRT
push to hub      ${HUB_USER:-<local only>}
filter           object > ${OBJECT_PENETRATION}m, self > ${SELF_PENETRATION}m$([[ $CENTER_SIGMA != 0 ]] && echo ", center sigma $CENTER_SIGMA")
EOF
[[ $DRY_RUN -eq 1 ]] && echo && echo "--dry-run: nothing below is executed"
echo

FAILED=()
TOTAL_START=$SECONDS

for obj in "${OBJECTS[@]}"; do
    MESH="$OBJECTS_DIR/${obj}_m.stl"
    NAME="${ROBOT}_grasp_${obj}"
    GEN="$OUT_ROOT/${NAME}_${VERSION}"
    RRT="$OUT_ROOT/${NAME}_rrt_${VERSION}"
    FILTERED="$OUT_ROOT/${NAME}_rrt_filtered_${VERSION}"

    echo "=== $obj ==============================================================="

    if has_stage gen; then
        echo "[generate] $GEN"
        if ! done_already "$GEN"; then
            run "${NAME}_${VERSION}_gen" \
                uv run python generate_dataset.py \
                    --robot "$ROBOT" \
                    --object_mesh_path "$MESH" \
                    --n_grasps "$N_GRASPS" \
                    --canonical_concentration "$CONCENTRATION" \
                    --output_dir "$GEN" \
                    --push_to_hub "$(hub "${NAME}_${VERSION}")" \
                || { FAILED+=("$obj/gen"); continue; }
        fi
    fi

    if has_stage rrt; then
        echo "[expand]   $RRT"
        if ! done_already "$RRT"; then
            run "${NAME}_${VERSION}_rrt" \
                uv run python grasp_rrt_expand.py \
                    --robot "$ROBOT" \
                    --object_mesh_path "$MESH" \
                    --n_grasps "$N_EXPANDED" \
                    --canonical_concentration "$CONCENTRATION" \
                    --dataset_path "$GEN" \
                    --output_dir "$RRT" \
                    --push_to_hub "$(hub "${NAME}_rrt_${VERSION}")" \
                || { FAILED+=("$obj/rrt"); continue; }
        fi
    fi

    if has_stage filter; then
        echo "[filter]   $FILTERED"
        if ! done_already "$FILTERED"; then
            CENTER_ARGS=()
            [[ $CENTER_SIGMA != 0 ]] && CENTER_ARGS=(--center_sigma "$CENTER_SIGMA")
            run "${NAME}_${VERSION}_filter" \
                uv run python filter_grasp_dataset.py \
                    --robot "$ROBOT" \
                    --object_mesh_path "$MESH" \
                    --dataset_path "$RRT" \
                    --collision_urdf "$DENSE_URDF" \
                    --max_object_penetration "$OBJECT_PENETRATION" \
                    --max_self_penetration "$SELF_PENETRATION" \
                    "${CENTER_ARGS[@]}" \
                    --output_dir "$FILTERED" \
                    --push_to_hub "$(hub "${NAME}_rrt_filtered_${VERSION}")" \
                || { FAILED+=("$obj/filter"); continue; }
        fi
    fi

    if has_stage plot; then
        # Plot whatever the latest existing stage produced.
        for candidate in "$FILTERED" "$RRT" "$GEN"; do
            [[ -d $candidate || $DRY_RUN -eq 1 ]] && { PLOT_SRC=$candidate; break; }
        done
        echo "[plot]     $PLOT_SRC"
        run "${NAME}_${VERSION}_plot" \
            uv run python plot_object_distribution.py \
                --robot "$ROBOT" --dataset_path "$PLOT_SRC" \
                --output "$OUT_ROOT/plots/$(basename "$PLOT_SRC").png" \
            || FAILED+=("$obj/plot")
    fi

    echo
done

# ---------------------------------------------------------------- summary
ELAPSED=$((SECONDS - TOTAL_START))
if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi
printf 'total %dh%02dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED: ${FAILED[*]}  (see $LOG_DIR)" >&2
    exit 1
fi
echo "all done -- inspect with:"
echo "  uv run python visualize_grasp.py --robot $ROBOT \\"
echo "    --object_mesh_path $OBJECTS_DIR/${OBJECTS[0]}_m.stl \\"
echo "    --dataset_path $OUT_ROOT/${ROBOT}_grasp_${OBJECTS[0]}_rrt_filtered_${VERSION}"

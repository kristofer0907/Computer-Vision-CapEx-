#!/bin/bash
# Claude Code statusline: model name w/ context window size (class-colored) + usage limit info (color-coded) + current directory (blue)

input=$(cat)

raw_cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd')
project_dir=$(echo "$input" | jq -r '.workspace.project_dir // empty')

# Show the current directory relative to the project root, e.g.
# "riact_riflex/common/src/" instead of the full absolute path.
# If cwd is outside/above the project root, fall back to just the
# directory name.
if [ -n "$project_dir" ] && [ "$project_dir" != "null" ]; then
  root_name=$(basename "$project_dir")
  case "$raw_cwd" in
    "$project_dir")
      cwd="${root_name}/"
      ;;
    "$project_dir"/*)
      rel="${raw_cwd#"$project_dir"/}"
      cwd="${root_name}/${rel}/"
      ;;
    *)
      cwd="$(basename "$raw_cwd")/"
      ;;
  esac
else
  cwd="$(basename "$raw_cwd")/"
fi

model_id=$(echo "$input" | jq -r '.model.id // empty')
model_display=$(echo "$input" | jq -r '.model.display_name // empty')
model_class=$(printf '%s %s' "$model_id" "$model_display" | grep -oiE 'haiku|sonnet|opus' | head -1 | tr '[:upper:]' '[:lower:]')

ctx_size=$(echo "$input" | jq -r '.context_window.context_window_size // empty')

# Build a human-readable model name, e.g. "claude-haiku-4-5-20251001" -> "Haiku 4.5"
nice_model=""
if [ -n "$model_id" ] && [ -n "$model_class" ]; then
  ver_raw=$(echo "$model_id" | sed -E "s/^claude-//; s/${model_class}-?//; s/-?[0-9]{8}$//; s/^-//; s/-$//")
  ver=$(echo "$ver_raw" | sed 's/-/./g')
  class_cap=$(echo "$model_class" | sed 's/^./\U&/')
  if [ -n "$ver" ]; then
    nice_model="$class_cap $ver"
  else
    nice_model="$class_cap"
  fi
elif [ -n "$model_display" ]; then
  nice_model="$model_display"
fi

# Append reasoning effort level next to the model name, e.g. "Sonnet 4.5 [High]"
effort=$(echo "$input" | jq -r '.effort.level // empty')
if [ -n "$effort" ] && [ "$effort" != "null" ]; then
  effort_cap=$(echo "$effort" | sed 's/^./\U&/')
  if [ -n "$nice_model" ]; then
    nice_model="$nice_model [$effort_cap]"
  fi
fi

# Format seconds-from-now as e.g. "2h15m" or "45m"
fmt_countdown() {
  awk -v secs="$1" 'BEGIN {
    if (secs < 0) secs = 0;
    h = int(secs / 3600);
    m = int((secs % 3600) / 60);
    if (h > 0) printf "%dh%dm", h, m;
    else printf "%dm", m;
  }'
}

# Time remaining until the 5h session rate limit resets (shown next to the 5h % below)
five_resets_at=$(echo "$input" | jq -r '.rate_limits.five_hour.resets_at // empty')
five_reset_str=""
if [ -n "$five_resets_at" ] && [ "$five_resets_at" != "null" ]; then
  now_ts=$(date +%s)
  delta=$((five_resets_at - now_ts))
  five_reset_str=$(fmt_countdown "$delta")
fi

# Fall back to the standard context window size for current Claude API models (200k) when not provided
if [ -z "$ctx_size" ] || [ "$ctx_size" = "null" ]; then
  ctx_size=200000
fi
ctx_fmt=$(awk -v n="$ctx_size" 'BEGIN {
  if (n >= 1000000) {
    v = n / 1000000;
    if (v == int(v)) printf "%dM", v; else printf "%.1fM", v;
  } else if (n >= 1000) {
    v = n / 1000;
    if (v == int(v)) printf "%dk", v; else printf "%.1fk", v;
  } else {
    printf "%d", n;
  }
}')

used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
five=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
week=$(echo "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')

# Returns a bold ANSI color code based on percentage thresholds:
# green (<50), yellow (50-75), red (>75)
pct_color() {
  awk -v p="$1" 'BEGIN {
    if (p > 75) print "\033[01;31m";
    else if (p >= 50) print "\033[01;33m";
    else print "\033[01;32m";
  }'
}

RESET='\033[00m'
DIR_COLOR='\033[01;34m'

# Distinct bold/vibrant color per model class
case "$model_class" in
  haiku)  MODEL_COLOR='\033[01;95m' ;; # bright magenta
  sonnet) MODEL_COLOR='\033[01;97m' ;; # bright white
  opus)   MODEL_COLOR='\033[01;93m' ;; # bright yellow
  *)      MODEL_COLOR='\033[01;96m' ;; # bright cyan (other models)
esac

info=""
if [ -n "$used" ]; then
  c=$(pct_color "$used")
  seg=$(printf "${c}Ctx: %.0f%%${RESET}" "$used")
  info="$seg"
fi
if [ -n "$five" ]; then
  c=$(pct_color "$five")
  if [ -n "$five_reset_str" ]; then
    seg=$(printf "${c}5h: %.0f%% (resets %s)${RESET}" "$five" "$five_reset_str")
  else
    seg=$(printf "${c}5h: %.0f%%${RESET}" "$five")
  fi
  info="${info:+$info | }$seg"
fi
if [ -n "$week" ]; then
  c=$(pct_color "$week")
  seg=$(printf "${c}7d: %.0f%%${RESET}" "$week")
  info="${info:+$info | }$seg"
fi

if [ -n "$nice_model" ]; then
  printf "${MODEL_COLOR}%s (%s)${RESET}" "$nice_model" "$ctx_fmt"
fi
if [ -n "$info" ]; then
  printf " (%s)" "$info"
fi
printf " ${DIR_COLOR}%s${RESET}" "$cwd"
printf '\n'

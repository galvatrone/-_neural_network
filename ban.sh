#!/usr/bin/env bash

set -u
shopt -s nullglob

clear


 text2="
██╗   ██╗███████╗ ██████╗ ████████╗ ██████╗ ██████╗
██║   ██║██╔════╝██╔════╝ ╚══██╔══╝██╔═══██╗██╔══██╗
██║   ██║█████╗  ██║         ██║   ██║   ██║██████╔╝
╚██╗ ██╔╝██╔══╝  ██║         ██║   ██║   ██║██╔══██╗
 ╚████╔╝ ███████╗╚██████╗    ██║   ╚██████╔╝██║  ██║
  ╚═══╝  ╚══════╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝

 "

 text="
 ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗███╗   ██╗███████╗████████╗
██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝████╗  ██║██╔════╝╚══██╔══╝
██║  ███╗███████║██║   ██║███████╗   ██║   ██╔██╗ ██║█████╗     ██║   
██║   ██║██╔══██║██║   ██║╚════██║   ██║   ██║╚██╗██║██╔══╝     ██║   
╚██████╔╝██║  ██║╚██████╔╝███████║   ██║   ██║ ╚████║███████╗   ██║   
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝   ╚═╝   
"

BLUE="\033[38;2;0;180;255m"
GREEN="\033[38;2;0;255;65m"
PURPLE="\033[38;2;190;60;255m"
YELLOW="\033[38;2;255;255;0m"
RED="\033[38;2;255;60;60m"
WHITE="\033[38;2;255;255;255m"
RESET="\033[0m"

lines=(
"$BLUE  ███████████████████████████████████████████████████████████████████████$RESET"
"$GREEN █           SYSTEM BOOTING          INITIALIZING CORE                █$RESET"
"$PURPLE█           GHOST PROTOCOL        ctOS TERMINAL  v2.7.4              █$RESET"
"$RED   ███████████████████████████████████████████████████████████████████████$RESET"
)

# Loading effect
for i in {0..20}; do
    printf "\r${GREEN}Loading: [%-20s] %d%%${RESET}" "$(printf "#%.0s" $(seq 1 "$i"))" $((i * 5))
    sleep 0.05
done
sleep 0.25
clear

# Flicker effect
for i in {1..5}; do
    echo -e "${RED}SYSTEM INITIALIZING...${RESET}"
    clear
    echo -e "${RED}$text${RESET}"
    sleep 0.02
    clear
    sleep 0.01
done

# Static glitch effect
for i in {1..12}; do
    clear
    echo -e "${RED}SYSTEM INITIALIZING...${RESET}"
    r=$((RANDOM % 4))
    echo -e "${lines[$r]}"
    echo
    echo -e "${GREEN}$text${RESET}"
    echo
    echo -e "${lines[$r]}"
    sleep 0.045
done

clear
echo -e "${GREEN}SYSTEM ONLINE${RESET}"
echo -e "${WHITE}███████████████████████████████████████████████████████████████████████${RESET}"
echo
echo -e "${WHITE}${text}${RESET}"
echo
echo -e "${WHITE}███████████████████████████████████████████████████████████████████████${RESET}"
echo

LINUX_CXX="g++"
WIN_CXX="x86_64-w64-mingw32-g++"
CPP_FLAGS=(-O2 -std=c++17)
WIN_FLAGS_METHOD1=(-static -static-libgcc -static-libstdc++ -s)
# Fallback: more aggressive static link mode.
WIN_FLAGS_METHOD2=(-static -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc -lwinpthread -Wl,-Bdynamic -lws2_32 -s)
BUILD_TARGET="${BAN_BUILD_TARGET:-}"
WIN_METHOD="${BAN_WIN_METHOD:-}"
RUN_AFTER="${BAN_RUN_AFTER:-0}"
RUN_FILE="${BAN_RUN_FILE:-}"
SOURCE_FILE="${BAN_SOURCE_FILE:-}"
TARGET_DIR_PRESET="${BAN_TARGET_DIR:-}"

cleanup_terminal() {
    tput cnorm 2>/dev/null || true
}
trap cleanup_terminal EXIT

choose_menu() {
    local title="$1"
    local default_index="$2"
    shift 2
    local options=("$@")
    local selected="$default_index"
    local count="${#options[@]}"
    local i key extra

    if [[ "$count" -eq 0 ]]; then
        MENU_SELECTED=0
        return
    fi

    # Fallback for non-interactive mode (no arrows available).
    if [[ ! -t 0 || ! -t 1 ]]; then
        MENU_SELECTED="$selected"
        return
    fi

    tput civis 2>/dev/null || true
    printf "%s\n" "$title"

    while true; do
        for ((i = 0; i < count; i++)); do
            if [[ "$i" -eq "$selected" ]]; then
                printf "  ${GREEN}> %s${RESET}\n" "${options[$i]}"
            else
                printf "    %s\n" "${options[$i]}"
            fi
        done

        IFS= read -rsn1 key
        if [[ "$key" == $'\x1b' ]]; then
            IFS= read -rsn2 extra || true
            key+="$extra"
        fi

        case "$key" in
            $'\x1b[A') selected=$(( (selected - 1 + count) % count )) ;;
            $'\x1b[B') selected=$(( (selected + 1) % count )) ;;
            "") break ;;
        esac

        printf "\033[%dA\r" "$count"
    done

    MENU_SELECTED="$selected"
    printf "\n"
}

platform_menu=("Linux" "Windows" "Both")
if [[ -z "$BUILD_TARGET" ]]; then
    choose_menu "Выбор платформы сборки (стрелки + Enter):" 2 "${platform_menu[@]}"
    case "$MENU_SELECTED" in
        0) BUILD_TARGET="linux" ;;
        1) BUILD_TARGET="windows" ;;
        *) BUILD_TARGET="both" ;;
    esac
fi

if [[ "$BUILD_TARGET" != "linux" && "$BUILD_TARGET" != "windows" && "$BUILD_TARGET" != "both" ]]; then
    echo -e "${RED}[ERROR]${RESET} Неверный BAN_BUILD_TARGET: $BUILD_TARGET"
    exit 1
fi

if [[ "$BUILD_TARGET" == "windows" || "$BUILD_TARGET" == "both" ]]; then
    if [[ -z "$WIN_METHOD" ]]; then
        win_menu=("Method 1 (обычный static)" "Method 2 (агрессивный static)" "Auto (1 -> fallback 2)")
        choose_menu "Выбор Windows метода (стрелки + Enter):" 2 "${win_menu[@]}"
        case "$MENU_SELECTED" in
            0) WIN_METHOD="1" ;;
            1) WIN_METHOD="2" ;;
            *) WIN_METHOD="auto" ;;
        esac
    fi
else
    WIN_METHOD="auto"
fi

if [[ "$WIN_METHOD" != "1" && "$WIN_METHOD" != "2" && "$WIN_METHOD" != "auto" ]]; then
    echo -e "${RED}[ERROR]${RESET} Неверный BAN_WIN_METHOD: $WIN_METHOD"
    exit 1
fi

if [[ "$BUILD_TARGET" == "linux" || "$BUILD_TARGET" == "both" ]]; then
    if ! command -v "$LINUX_CXX" >/dev/null 2>&1; then
        echo -e "${RED}[ERROR]${RESET} g++ не найден."
        exit 1
    fi
fi

if [[ "$BUILD_TARGET" == "windows" || "$BUILD_TARGET" == "both" ]]; then
    if ! command -v "$WIN_CXX" >/dev/null 2>&1; then
        echo -e "${RED}[ERROR]${RESET} $WIN_CXX не найден."
        exit 1
    fi
fi

TARGET_DIR="${1:-}"
if [[ -z "$TARGET_DIR" ]]; then
    if [[ -n "$TARGET_DIR_PRESET" ]]; then
        TARGET_DIR="$TARGET_DIR_PRESET"
    else
        read -r -p "Папка для компиляции (. по умолчанию): " TARGET_DIR
    fi
fi

TARGET_DIR="${TARGET_DIR:-.}"
TARGET_DIR="${TARGET_DIR%/}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo -e "${RED}[ERROR]${RESET} Папка '$TARGET_DIR' не найдена."
    exit 1
fi

echo -e "${GREEN}[BUILD]${RESET} ${WHITE}Целевая папка ->${RESET} $TARGET_DIR"
echo -e "${GREEN}[BUILD]${RESET} ${WHITE}Режим платформы ->${RESET} $BUILD_TARGET"
if [[ "$BUILD_TARGET" == "windows" || "$BUILD_TARGET" == "both" ]]; then
    echo -e "${GREEN}[BUILD]${RESET} ${WHITE}Windows method ->${RESET} $WIN_METHOD"
fi

sources=("$TARGET_DIR"/*.cpp)
if [[ -n "$SOURCE_FILE" ]]; then
    if [[ -f "$SOURCE_FILE" && "${SOURCE_FILE##*.}" == "cpp" ]]; then
        sources=("$SOURCE_FILE")
    else
        echo -e "${YELLOW}[WARN]${RESET} BAN_SOURCE_FILE не .cpp или файл не найден, использую .cpp из '$TARGET_DIR'."
        if [[ -f "$TARGET_DIR/main.cpp" ]]; then
            sources=("$TARGET_DIR/main.cpp")
        fi
    fi
fi
if [[ ${#sources[@]} -eq 0 ]]; then
    echo -e "${YELLOW}[WARN]${RESET} В папке '$TARGET_DIR' нет .cpp файлов."
    exit 0
fi

fail=0

for src in "${sources[@]}"; do
    base="${src%.cpp}"

    if [[ "$BUILD_TARGET" == "linux" || "$BUILD_TARGET" == "both" ]]; then
        echo -e "${GREEN}[BUILD]${RESET} Linux: $src -> $base"
        if ! "$LINUX_CXX" "${CPP_FLAGS[@]}" "$src" -o "$base"; then
            echo -e "${RED}[ERROR]${RESET} Не удалось собрать Linux бинарник для $src"
            fail=1
            continue
        fi
    fi

    if [[ "$BUILD_TARGET" == "windows" || "$BUILD_TARGET" == "both" ]]; then
        if [[ "$WIN_METHOD" == "1" ]]; then
            echo -e "${GREEN}[BUILD]${RESET} Windows (method 1): $src -> ${base}.exe"
            if ! "$WIN_CXX" "${CPP_FLAGS[@]}" "$src" -o "${base}.exe" "${WIN_FLAGS_METHOD1[@]}"; then
                echo -e "${RED}[ERROR]${RESET} Не удалось собрать Windows бинарник для $src (method 1)."
                fail=1
            fi
        elif [[ "$WIN_METHOD" == "2" ]]; then
            echo -e "${GREEN}[BUILD]${RESET} Windows (method 2): $src -> ${base}.exe"
            if ! "$WIN_CXX" "${CPP_FLAGS[@]}" "$src" -o "${base}.exe" "${WIN_FLAGS_METHOD2[@]}"; then
                echo -e "${RED}[ERROR]${RESET} Не удалось собрать Windows бинарник для $src (method 2)."
                fail=1
            fi
        else
            echo -e "${GREEN}[BUILD]${RESET} Windows (method 1): $src -> ${base}.exe"
            if ! "$WIN_CXX" "${CPP_FLAGS[@]}" "$src" -o "${base}.exe" "${WIN_FLAGS_METHOD1[@]}" >/tmp/win_build_err_1.log 2>&1; then
                echo -e "${YELLOW}[WARN]${RESET} Method 1 не прошел. Пробую method 2 (full static fallback)..."
                if ! "$WIN_CXX" "${CPP_FLAGS[@]}" "$src" -o "${base}.exe" "${WIN_FLAGS_METHOD2[@]}" >/tmp/win_build_err_2.log 2>&1; then
                    echo -e "${RED}[ERROR]${RESET} Не удалось собрать Windows бинарник для $src (оба метода)."
                    echo -e "${YELLOW}[INFO]${RESET} Ошибка method 1:"
                    cat /tmp/win_build_err_1.log
                    echo -e "${YELLOW}[INFO]${RESET} Ошибка method 2:"
                    cat /tmp/win_build_err_2.log
                    fail=1
                else
                    echo -e "${GREEN}[OK]${RESET} Windows build собран методом 2."
                fi
            fi
        fi        
    fi
done

if [[ "$fail" -ne 0 ]]; then
    exit 1
fi

if [[ "$RUN_AFTER" == "1" ]]; then
    run_target="$RUN_FILE"
    if [[ -n "$run_target" && "$run_target" != */* ]]; then
        run_target="$TARGET_DIR/$run_target"
    fi

    if [[ "$BUILD_TARGET" == "windows" && -n "$run_target" ]]; then
        run_target="${run_target%.exe}.exe"
        echo -e "${YELLOW}[WARN]${RESET} Автозапуск .exe на Linux может не работать без wine."
    fi

    # If preset run file is wrong, fallback to the actual single built source.
    if [[ -n "$run_target" && ! -e "$run_target" && ${#sources[@]} -eq 1 ]]; then
        fallback_target="${sources[0]%.cpp}"
        if [[ "$BUILD_TARGET" == "windows" ]]; then
            fallback_target="${fallback_target}.exe"
        fi
        echo -e "${YELLOW}[WARN]${RESET} RUN файл не найден, использую собранный: $fallback_target"
        run_target="$fallback_target"
    elif [[ -z "$run_target" && ${#sources[@]} -eq 1 ]]; then
        run_target="${sources[0]%.cpp}"
        if [[ "$BUILD_TARGET" == "windows" ]]; then
            run_target="${run_target}.exe"
        fi
    fi

    if [[ -n "$run_target" ]]; then
        echo -e "${GREEN}[RUN]${RESET} ${WHITE}Запуск ->${RESET} $run_target"
        if [[ -x "$run_target" || "$run_target" == *.exe ]]; then
            "$run_target"
        else
            echo -e "${YELLOW}[WARN]${RESET} Файл не найден или не исполняемый: $run_target"
        fi
    else        
        echo -e "${YELLOW}[WARN]${RESET} Не удалось определить файл для автозапуска."
    fi
fi

echo -e "${GREEN}[OK]${RESET} Готово."

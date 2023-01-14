#include <stdlib.h>

#define USE_EXPLORATION_BY_KEYS
#define IGFD_KEY_UP ImGuiKey_UpArrow
#define IGFD_KEY_DOWN ImGuiKey_DownArrow
#define IGFD_KEY_ENTER ImGuiKey_Enter
#define IGFD_KEY_BACKSPACE ImGuiKey_Backspace
#define USE_DIALOG_EXIT_WITH_KEY
#define IGFD_EXIT_KEY ImGuiKey_Escape
#define FILTER_COMBO_WIDTH 200.0f
#define dirNameString "Directory Path:"
#define OverWriteDialogTitleString "The file already exists!"
#define OverWriteDialogMessageString "Would you like to overwrite it?"
#define okButtonWidth 100.0f
#define cancelButtonWidth 100.0f
#define fileSizeBytes "B"
#define fileSizeKiloBytes "KiB"
#define fileSizeMegaBytes "MiB"
#define fileSizeGigaBytes "GiB"

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include "cimgui.h"
#include "ImGuiFileDialog.h"

if(HIP_POT_COMPACT_MEM_LAYOUT_FLAG)
        set(HIP_POT_COMPACT_MEM_LAYOUT ON)
endif()
if(HIP_POT_DEVICE_API_INLINE_FLAG)
        set(HIP_POT_DEVICE_API_INLINE ON)
endif()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/pot_building_config.h.in"
        "${CONFIGURE_GENERATED_PATH}/pot_building_config.h"
)

install(FILES "${CONFIGURE_GENERATED_PATH}/pot_building_config.h"
        DESTINATION "include"
        )

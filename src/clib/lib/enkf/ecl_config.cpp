#include <filesystem>
#include <unordered_map>

#include <stdlib.h>
#include <time.h>

#include <ert/res_util/ui_return.hpp>
#include <ert/util/parser.h>
#include <ert/util/util.h>

#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_io_config.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/enkf_defaults.hpp>

namespace fs = std::filesystem;

/**
 The ecl_config_struct holds configuration information needed to run ECLIPSE.

 Pointers to the fields in this structure are passed on to e.g. the
 enkf_state->shared_info object, but this struct is the *OWNER* of
 this information, and hence responsible for booting and deleting
 these objects.

 Observe that the distinction of what goes in model_config, and what
 goes in ecl_config is not entirely clear.
 */
struct ecl_config_struct {
    /** This struct contains information of whether the eclipse files should be
     * formatted|unified|endian_fliped */
    ecl_io_config_type *io_config;
    /** Eclipse data file. */
    char *data_file;
    /** An optional date value which can be used to check if the ECLIPSE
     * simulation has been 'long enough'. */
    ecl_sum_type *refcase;
    const char *refcase_name;
    /** The grid which is active for this model. */
    ecl_grid_type *grid;
    /** Name of schedule prediction file - observe that this is internally
     * handled as a gen_kw node. */
    char *schedule_prediction_file;
    int last_history_restart;
    /** Have we found the <INIT> tag in the data file? */
    bool can_restart;
    bool have_eclbase;
    /** We should parse the ECLIPSE data file and determine how many cpus this
     * eclipse file needs. */
    int num_cpu;
    /** Either metric, field or lab */
    ert_ecl_unit_enum unit_system;
};

/**
 With this function we try to determine whether ECLIPSE is active
 for this case, i.e. if ECLIPSE is part of the forward model. This
 should ideally be inferred from the FORWARD model, but what we do
 here is just to check if the core field ->eclbase or ->data_file
 have been set. If they are both equal to NULL we assume that
 ECLIPSE is not active and return false, otherwise we return true.
 */
bool ecl_config_active(const ecl_config_type *config) {
    if (config->have_eclbase)
        return true;

    if (config->data_file)
        return true;

    return false;
}

/*
   Could look up the sched_file instance directly - because the
   ecl_config will never be the owner of a file with predictions.
   */

int ecl_config_get_last_history_restart(const ecl_config_type *ecl_config) {
    return ecl_config->last_history_restart;
}

bool ecl_config_can_restart(const ecl_config_type *ecl_config) {
    return ecl_config->can_restart;
}

void ecl_config_assert_restart(const ecl_config_type *ecl_config) {
    if (!ecl_config_can_restart(ecl_config)) {
        fprintf(stderr, "** Warning - tried to restart case which is not "
                        "properly set up for restart.\n");
        fprintf(stderr, "** Need <INIT> in datafile and INIT_SECTION keyword "
                        "in config file.\n");
        util_exit("%s: exiting \n", __func__);
    }
}

ui_return_type *ecl_config_validate_data_file(const ecl_config_type *ecl_config,
                                              const char *data_file) {
    if (fs::exists(data_file))
        return ui_return_alloc(UI_RETURN_OK);
    else {
        ui_return_type *ui_return = ui_return_alloc(UI_RETURN_FAIL);
        char *error_msg = util_alloc_sprintf("File not found:%s", data_file);
        ui_return_add_error(ui_return, error_msg);
        free(error_msg);
        return ui_return;
    }
}

void ecl_config_set_data_file(ecl_config_type *ecl_config,
                              const char *data_file) {
    ecl_config->data_file =
        util_realloc_string_copy(ecl_config->data_file, data_file);
    {
        FILE *stream = util_fopen(ecl_config->data_file, "r");
        basic_parser_type *parser =
            basic_parser_alloc(NULL, NULL, NULL, NULL, "--", "\n");
        const char *init_tag = DEFAULT_START_TAG "INIT" DEFAULT_END_TAG;

        ecl_config->can_restart =
            basic_parser_fseek_string(parser, stream, init_tag, false, true);

        basic_parser_free(parser);
        fclose(stream);
    }
    ecl_config->num_cpu = ecl_util_get_num_cpu(ecl_config->data_file);
    ecl_config->unit_system = ecl_util_get_unit_set(ecl_config->data_file);
}

const char *ecl_config_get_data_file(const ecl_config_type *ecl_config) {
    return ecl_config->data_file;
}

int ecl_config_get_num_cpu(const ecl_config_type *ecl_config) {
    return ecl_config->num_cpu;
}

const char *
ecl_config_get_schedule_prediction_file(const ecl_config_type *ecl_config) {
    return ecl_config->schedule_prediction_file;
}

/**
   Observe: The real schedule prediction functionality is implemented
   as a special GEN_KW node in ensemble_config.
 */
void ecl_config_set_schedule_prediction_file(
    ecl_config_type *ecl_config, const char *schedule_prediction_file) {
    ecl_config->schedule_prediction_file = util_realloc_string_copy(
        ecl_config->schedule_prediction_file, schedule_prediction_file);
}

ui_return_type *ecl_config_validate_eclbase(const ecl_config_type *ecl_config,
                                            const char *eclbase_fmt) {
    if (ecl_util_valid_basename_fmt(eclbase_fmt))
        return ui_return_alloc(UI_RETURN_OK);
    else {
        ui_return_type *ui_return = ui_return_alloc(UI_RETURN_FAIL);
        {
            char *error_msg = util_alloc_sprintf(
                "The format string: %s was invalid as ECLBASE format",
                eclbase_fmt);
            ui_return_add_error(ui_return, error_msg);
            free(error_msg);
        }
        ui_return_add_help(
            ui_return,
            "The eclbase format must have all characters in the same case,");
        ui_return_add_help(
            ui_return,
            "in addition it can contain a %d specifier which will be");
        ui_return_add_help(ui_return, "with the realization number.");

        return ui_return;
    }
}

/**
 Can be called with @refcase == NULL - which amounts to clearing the
 current refcase.
*/
bool ecl_config_load_refcase(ecl_config_type *ecl_config,
                             const char *case_name) {
    if (case_name) {
        auto ecl_sum = ecl_sum_fread_alloc_case(case_name, ":");
        if (ecl_sum == NULL) {
            ecl_config->refcase = NULL;
            return false;
        }
        ecl_config->refcase = ecl_sum;
        return true;
    } else {
        ecl_config->refcase = NULL;
        return true;
    }
}

ui_return_type *ecl_config_validate_refcase(const ecl_config_type *ecl_config,
                                            const char *refcase) {
    if (ecl_sum_case_exists(refcase))
        return ui_return_alloc(UI_RETURN_OK);
    else {
        ui_return_type *ui_return = ui_return_alloc(UI_RETURN_FAIL);
        char *error_msg = util_alloc_sprintf(
            "Could not load summary case from:%s \n", refcase);
        ui_return_add_error(ui_return, error_msg);
        free(error_msg);
        return ui_return;
    }
}

/**
 Will return NULL if no refcase is set.
*/
const char *ecl_config_get_refcase_name(const ecl_config_type *ecl_config) {
    auto refcase = ecl_config->refcase;
    if (!refcase)
        return NULL;
    return ecl_sum_get_case(ecl_config->refcase);
}

static ecl_config_type *ecl_config_alloc_empty(void) {
    ecl_config_type *ecl_config = new ecl_config_type();

    ecl_config->io_config = ecl_io_config_alloc(
        DEFAULT_FORMATTED, DEFAULT_UNIFIED, DEFAULT_UNIFIED);
    ecl_config->have_eclbase = false;
    ecl_config->num_cpu =
        1; /* This must get a valid default in case no ECLIPSE datafile is provided. */
    ecl_config->unit_system = ECL_METRIC_UNITS;
    ecl_config->data_file = NULL;
    ecl_config->grid = NULL;
    ecl_config->can_restart = false;
    ecl_config->schedule_prediction_file = NULL;
    ecl_config->refcase = NULL;

    return ecl_config;
}

ecl_config_type *ecl_config_alloc(const config_content_type *config_content) {
    ecl_config_type *ecl_config = ecl_config_alloc_empty();

    if (config_content)
        ecl_config_init(ecl_config, config_content);

    return ecl_config;
}

ecl_config_type *ecl_config_alloc_full(bool have_eclbase, char *data_file,
                                       ecl_grid_type *grid,
                                       char *refcase_default,
                                       char *sched_prediction_file) {
    ecl_config_type *ecl_config = ecl_config_alloc_empty();
    ecl_config->have_eclbase = have_eclbase;
    ecl_config->grid = grid;
    if (data_file != NULL) {
        ecl_config_set_data_file(ecl_config, data_file);
    }

    if (refcase_default)
        if (!ecl_config_load_refcase(ecl_config, refcase_default))
            fprintf(stderr, "** Warning: loading refcase:%s failed \n",
                    refcase_default);

    if (sched_prediction_file)
        ecl_config->schedule_prediction_file =
            util_alloc_string_copy(sched_prediction_file);

    return ecl_config;
}

static void handle_has_eclbase_key(ecl_config_type *ecl_config,
                                   const config_content_type *config) {
    /*
     The eclbase is not internalized here; here we only flag that the
     ECLBASE keyword has been present in the configuration. The
     actualt value is internalized as a job_name in the model_config.
   */

    if (config_content_has_item(config, ECLBASE_KEY)) {
        ui_return_type *ui_return = ecl_config_validate_eclbase(
            ecl_config, config_content_iget(config, ECLBASE_KEY, 0, 0));
        if (ui_return_get_status(ui_return) == UI_RETURN_OK)
            ecl_config->have_eclbase = true;
        else
            util_abort("%s: failed to set eclbase format. Error:%s\n", __func__,
                       ui_return_get_last_error(ui_return));
        ui_return_free(ui_return);
    }
}

static void handle_has_data_file_key(ecl_config_type *ecl_config,
                                     const config_content_type *config) {
    const char *data_file =
        config_content_get_value_as_abspath(config, DATA_FILE_KEY);
    ui_return_type *ui_return =
        ecl_config_validate_data_file(ecl_config, data_file);
    if (ui_return_get_status(ui_return) == UI_RETURN_OK)
        ecl_config_set_data_file(ecl_config, data_file);
    else
        util_abort("%s: problem setting ECLIPSE data file (%s)\n", __func__,
                   ui_return_get_last_error(ui_return));
    ui_return_free(ui_return);
}

static void handle_has_grid_key(ecl_config_type *ecl_config,
                                const config_content_type *config) {
    const char *grid_file =
        config_content_get_value_as_abspath(config, GRID_KEY);

    ui_return_type *ui_return = ecl_config_validate_grid(ecl_config, grid_file);
    if (ui_return_get_status(ui_return) == UI_RETURN_OK)
        ecl_config_set_grid(ecl_config, grid_file);
    else
        util_abort("%s: failed to set grid file:%s  Error:%s \n", __func__,
                   grid_file, ui_return_get_last_error(ui_return));

    ui_return_free(ui_return);
}

static void handle_has_refcase_key(ecl_config_type *ecl_config,
                                   const config_content_type *config) {
    const char *refcase_path =
        config_content_get_value_as_abspath(config, REFCASE_KEY);

    if (!ecl_config_load_refcase(ecl_config, refcase_path))
        fprintf(stderr, "** Warning: loading refcase:%s failed \n",
                refcase_path);
}

static void
handle_has_schedule_prediction_file_key(ecl_config_type *ecl_config,
                                        const config_content_type *config) {
    const config_content_item_type *pred_item =
        config_content_get_item(config, SCHEDULE_PREDICTION_FILE_KEY);

    config_content_node_type *pred_node =
        config_content_item_get_last_node(pred_item);
    const char *template_file = config_content_node_iget_as_path(pred_node, 0);
    ecl_config_set_schedule_prediction_file(ecl_config, template_file);
}

void ecl_config_init(ecl_config_type *ecl_config,
                     const config_content_type *config) {
    if (config_content_has_item(config, ECLBASE_KEY))
        handle_has_eclbase_key(ecl_config, config);

    if (config_content_has_item(config, DATA_FILE_KEY))
        handle_has_data_file_key(ecl_config, config);

    if (config_content_has_item(config, GRID_KEY))
        handle_has_grid_key(ecl_config, config);

    if (config_content_has_item(config, REFCASE_KEY))
        handle_has_refcase_key(ecl_config, config);

    if (ecl_config->can_restart)
        fprintf(
            stderr,
            "** Warning: The ECLIPSE data file contains a <INIT> section, the\n"
            "            support for this functionality has been removed. Ert\n"
            "            will not be able to properly initialize the ECLIPSE "
            "MODEL.\n");

    if (config_content_has_item(config, SCHEDULE_PREDICTION_FILE_KEY))
        handle_has_schedule_prediction_file_key(ecl_config, config);
}

void ecl_config_free(ecl_config_type *ecl_config) {
    ecl_io_config_free(ecl_config->io_config);
    free(ecl_config->data_file);
    free(ecl_config->schedule_prediction_file);

    if (ecl_config->grid != NULL)
        ecl_grid_free(ecl_config->grid);

    if (ecl_config->refcase)
        ecl_sum_free(ecl_config->refcase);

    delete ecl_config;
}

ecl_grid_type *ecl_config_get_grid(const ecl_config_type *ecl_config) {
    return ecl_config->grid;
}

const char *ecl_config_get_gridfile(const ecl_config_type *ecl_config) {
    if (ecl_config->grid == NULL)
        return NULL;
    else
        return ecl_grid_get_name(ecl_config->grid);
}

/**
   The ecl_config object isolated supports run-time changing of the
   grid, however this does not (in general) apply to the system as a
   whole. Other objects which internalize pointers (i.e. field_config
   objects) to an ecl_grid_type instance will be left with dangling
   pointers; and things will probably die an ugly death. So - changing
   grid runtime should be done with extreme care.
*/
void ecl_config_set_grid(ecl_config_type *ecl_config, const char *grid_file) {
    if (ecl_config->grid != NULL)
        ecl_grid_free(ecl_config->grid);
    ecl_config->grid = ecl_grid_alloc(grid_file);
}

ui_return_type *ecl_config_validate_grid(const ecl_config_type *ecl_config,
                                         const char *grid_file) {
    ui_return_type *ui_return;
    if (fs::exists(grid_file)) {
        ecl_file_enum file_type = ecl_util_get_file_type(grid_file, NULL, NULL);
        if ((file_type == ECL_EGRID_FILE) || (file_type == ECL_GRID_FILE))
            ui_return = ui_return_alloc(UI_RETURN_OK);
        else {
            ui_return = ui_return_alloc(UI_RETURN_FAIL);
            ui_return_add_error(ui_return,
                                "Input argument is not a GRID/EGRID file");
        }
    } else {
        ui_return = ui_return_alloc(UI_RETURN_FAIL);
        ui_return_add_error(ui_return, "Input argument does not exist.");
    }
    return ui_return;
}

const ecl_sum_type *ecl_config_get_refcase(const ecl_config_type *ecl_config) {
    return ecl_config->refcase;
}

bool ecl_config_has_refcase(const ecl_config_type *ecl_config) {
    const ecl_sum_type *refcase = ecl_config_get_refcase(ecl_config);
    if (refcase)
        return true;
    else
        return false;
}

bool ecl_config_get_formatted(const ecl_config_type *ecl_config) {
    return ecl_io_config_get_formatted(ecl_config->io_config);
}
bool ecl_config_get_unified_restart(const ecl_config_type *ecl_config) {
    return ecl_io_config_get_unified_restart(ecl_config->io_config);
}

bool ecl_config_have_eclbase(const ecl_config_type *ecl_config) {
    return ecl_config->have_eclbase;
}

/** Units as specified in the ECLIPSE technical manual */
const char *ecl_config_get_depth_unit(const ecl_config_type *ecl_config) {
    switch (ecl_config->unit_system) {
    case ECL_METRIC_UNITS:
        return "M";
    case ECL_FIELD_UNITS:
        return "FT";
    case ECL_LAB_UNITS:
        return "CM";
    default:
        util_abort("%s: unit system enum value:%d not recognized \n", __func__,
                   ecl_config->unit_system);
        return NULL;
    }
}

const char *ecl_config_get_pressure_unit(const ecl_config_type *ecl_config) {
    switch (ecl_config->unit_system) {
    case ECL_METRIC_UNITS:
        return "BARSA";
    case ECL_FIELD_UNITS:
        return "PSIA";
    case ECL_LAB_UNITS:
        return "ATMA";
    default:
        util_abort("%s: unit system enum value:%d not recognized \n", __func__,
                   ecl_config->unit_system);
        return NULL;
    }
}

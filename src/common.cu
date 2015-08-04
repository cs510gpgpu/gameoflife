#include "common.h"
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include "argtable3.h"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct arg_lit *help, *profile;
struct arg_int *height, *width, *block_width;
struct arg_str *mode;
struct arg_end *end;

int processArgs(const char * progname, char ** argv, int argc, MODES * m, int * h, int * w, int * bw, int * p)
{
    int do_exit = 0;
    void *argtable[] = {
        help    = arg_litn(NULL, "help",  0, 1, "display this help and exit"),       
        height  = arg_intn(NULL, "height","<n>", 0, 1,   "height"),
        width   = arg_intn(NULL, "width", "<n>",  0, 1, "width"),
        block_width = arg_intn(NULL, "blockwidth", "<n>",  0, 1, "block width"),
        mode    = arg_strn(NULL, "mode", "modename",  0, 1, "allowed modes: gpu cpu"),
        profile = arg_litn(NULL, "profile", 0, 1, "disables text output"),
        end     = arg_end(20),
    };

    int exitcode = 0;
    int nerrors;
    nerrors = arg_parse(argc,argv,argtable);

    /* special case: '--help' takes precedence over error reporting */
    if (help->count > 0)
    {
        printf("Usage: %s", progname);
        arg_print_syntax(stdout, argtable, "\n");
        arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        exitcode = 0;
        do_exit = 1;
        goto exit;
    }

    /* If the parser returned any errors then display them and exit */
    if (nerrors > 0)
    {
        /* Display the error details contained in the arg_end struct.*/
        arg_print_errors(stdout, end, progname);
        printf("Try '%s --help' for more information.\n", progname);
        exitcode = 1;
        do_exit = 1;
        goto exit;
    }

    if (height->count > 0) {
        *h = height->ival[0];
    }
    if (width->count > 0) {
        *w = width->ival[0];
    }

    if (block_width->count > 0) {
        *bw = block_width->ival[0];
    }
    
    if (mode->count == 0) {
        *m = PROFILE_NONE;
    } else if (!strcmp(mode->sval[0], "gpu")) {
        *m = PROFILE_GPU;
    } else if (!strcmp(mode->sval[0], "cpu")) {
        *m = PROFILE_CPU;
    } else {
        printf("Unknown mode type \"%s\" \n", mode->sval[0]);
        do_exit = 1;
        exitcode = 1;
        goto exit;
    }

    *p = profile->count > 0;
exit:
    /* deallocate each non-null entry in argtable[] */
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
    if (do_exit) {
        exit(exitcode);
    }
    return exitcode;
}

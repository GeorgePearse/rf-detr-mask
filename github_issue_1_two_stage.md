# Issue: Removed two_stage parameter from codebase but still exists in configs

## Description
The recent commit `17726e842a65062c4475b67ad6aff623406328d2` removed the `two_stage` parameter from the codebase, including from the transformer implementation and config file definitions. However, there is still a reference to this parameter in at least one configuration file: `/configs/1`.

## Impact
This discrepancy creates confusion and inconsistency in the codebase. The code no longer uses the `two_stage` parameter, but configurations may still include it, which could cause unexpected behavior or confusion for contributors and users.

## Steps to reproduce
1. Check the configuration file at `/configs/1` which still includes `two_stage: true` in the model section
2. Try loading this configuration, which would contain an unused parameter

## Proposed solution
1. Remove the `two_stage` parameter from all remaining configuration files
2. Update documentation to clarify that the two-stage mode has been removed from the codebase
3. Consider adding validation in the configuration loading to warn about deprecated parameters

## Additional context
The `two_stage` parameter appears to have been removed because the functionality was made default, with the transformer architecture now always utilizing this behavior. Maintaining this consistency across all configurations would improve code clarity.
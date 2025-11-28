import os

def check_paths(*paths: str) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def iterate_conditions(groups, tasks, task_stages, subjects=None):
    """
    Yield all valid combinations of (group, task, task_stage, block, subject).

    If subjects is None, yield 4-tuples:
        (group, task, task_stage, block)

    If subjects is provided, yield 5-tuples:
        (group, task, task_stage, block, subject)
    """
    for group in groups:
        for task in tasks:
            # determine blocks
            blocks = [''] if task == '_BL' else ['_baseline', '_adaptation']

            for task_stage in task_stages:
                for block in blocks:
                    if subjects is None:
                        # no subject-level iteration
                        yield group, task, task_stage, block
                    else:
                        for subject in subjects:
                            yield group, task, task_stage, block, subject

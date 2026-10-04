# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: FORCE

default: FORCE
	@just

git-init: FORCE
	@just git-init

git-remote: FORCE
	@just git-remote '$(REMOTE_URL)'

Makefile: ;

%: FORCE
	@just $@

FORCE: ;

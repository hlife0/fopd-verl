# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU regressions for follow waiting and progress notifications."""

import asyncio

import pytest

from verl.experimental.teacher_loop.teacher_follow import StudentTokenState, follow_submit_len
from verl.experimental.teacher_loop.teacher_manager import _wait_for_follow_progress


def test_already_submittable_progress_does_not_wait():
    """A previously delivered notification must not be needed after progress is visible."""

    async def run():
        state = StudentTokenState()
        state.set_prompt(list(range(17)))
        await asyncio.wait_for(_wait_for_follow_progress(state, 0), timeout=1)

    asyncio.run(run())


def test_stale_notification_waits_for_new_progress():
    """Clear the prompt notification and remain waiting until another update arrives."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1])
        waiter = asyncio.create_task(_wait_for_follow_progress(state, 0))
        try:
            await asyncio.sleep(0)
            assert not waiter.done()
            state.update_response(list(range(16)))
            await asyncio.wait_for(waiter, timeout=1)
            assert follow_submit_len(len(state.snapshot()), state.student_done, 0) == 17
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("already_done", [False, True])
def test_eos_releases_wait_for_short_tail(already_done):
    """Completion releases a short tail, even if completion preceded the wait."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1])
        state.update_response([2, 3])
        if already_done:
            state.mark_done()
        waiter = asyncio.create_task(_wait_for_follow_progress(state, 0))
        try:
            await asyncio.sleep(0)
            if not already_done:
                assert not waiter.done()
                state.mark_done()
            await asyncio.wait_for(waiter, timeout=1)
            assert follow_submit_len(len(state.snapshot()), state.student_done, 0) == 3
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    asyncio.run(run())


def test_eos_without_new_tokens_does_not_wait():
    """A completed sequence with no remaining submission must terminate its loop."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1, 2, 3])
        state.mark_done()
        await asyncio.wait_for(_wait_for_follow_progress(state, 3), timeout=1)
        assert follow_submit_len(len(state.snapshot()), state.student_done, 3) == 0

    asyncio.run(run())


def test_small_update_wakes_but_does_not_force_submission():
    """The session must recheck eligibility after a notification below the block threshold."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1])
        first = asyncio.create_task(_wait_for_follow_progress(state, 0))
        second = None
        try:
            await asyncio.sleep(0)
            state.update_response([2])
            await asyncio.wait_for(first, timeout=1)
            assert follow_submit_len(len(state.snapshot()), state.student_done, 0) == 0
            second = asyncio.create_task(_wait_for_follow_progress(state, 0))
            await asyncio.sleep(0)
            assert not second.done()
            state.mark_done()
            await asyncio.wait_for(second, timeout=1)
        finally:
            for task in (first, second):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_progress_arriving_at_clear_is_rechecked():
    """Progress visible immediately after clearing the notification is not lost."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1])

        class UpdatingEvent(asyncio.Event):
            def clear(self):
                super().clear()
                state.update_response(list(range(16)))
                super().clear()

        state.event = UpdatingEvent()
        await asyncio.wait_for(_wait_for_follow_progress(state, 0), timeout=1)

    asyncio.run(run())


def test_cancelled_wait_does_not_consume_future_progress():
    """Cancelling one wait must leave later progress observable by another wait."""

    async def run():
        state = StudentTokenState()
        state.set_prompt([1])
        waiter = asyncio.create_task(_wait_for_follow_progress(state, 0))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        state.update_response(list(range(16)))
        await asyncio.wait_for(_wait_for_follow_progress(state, 0), timeout=1)

    asyncio.run(run())



import asyncio

import pytest


@pytest.fixture
async def aiohttp_server_factory():
    """Serve a fixed body over real HTTP, streamed in small chunks."""
    from aiohttp import web

    runners = []

    async def factory(payload: bytes, chunk_delay: float = 0.02) -> str:
        async def handler(request):
            resp = web.StreamResponse(
                status=200, headers={'Content-Length': str(len(payload)), 'Content-Type': 'application/octet-stream'}
            )
            await resp.prepare(request)
            for i in range(0, len(payload), 1024):
                await resp.write(payload[i : i + 1024])
                # Force the body to arrive over several event-loop passes so a
                # reader that takes only what is buffered truncates the body.
                if chunk_delay:
                    await asyncio.sleep(chunk_delay)
            await resp.write_eof()
            return resp

        app = web.Application()
        app.router.add_get('/media', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        runners.append(runner)
        port = runner.addresses[0][1]
        return f'http://127.0.0.1:{port}/media'

    yield factory
    for r in runners:
        await r.cleanup()

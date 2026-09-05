// CB16 CI Worker Edge Auth (Cloudflare Access-equivalent for V1.1)
// Edge layer validates only the Cloudflare Service Token.
// CB16 application Bearer is validated by the OCI FastAPI relay.
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (!url.pathname.startsWith('/api/v1/worker/')) {
      return fetch(request);
    }

    const clientId = request.headers.get('CF-Access-Client-Id') || '';
    const clientSecret = request.headers.get('CF-Access-Client-Secret') || '';

    if (
      clientId !== env.CF_CLIENT_ID ||
      clientSecret !== env.CF_CLIENT_SECRET
    ) {
      return new Response(JSON.stringify({ error: 'unauthorized' }), {
        status: 403,
        headers: { 'content-type': 'application/json' },
      });
    }

    const internalUrl = new URL(request.url);
    internalUrl.pathname = '/cb16-internal' + url.pathname;
    internalUrl.search = url.search;

    const headers = new Headers(request.headers);
    headers.set('X-CB16-Internal-Secret', env.CB16_INTERNAL_SECRET);

    return fetch(internalUrl.toString(), {
      method: request.method,
      headers,
      body: request.body,
      redirect: 'manual',
    });
  },
};

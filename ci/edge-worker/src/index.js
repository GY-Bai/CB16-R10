// CB16 CI Worker Edge Auth
// Approximates Cloudflare Access Service Auth at the edge.
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (!url.pathname.startsWith('/api/v1/worker/')) {
      return fetch(request);
    }

    const clientId = request.headers.get('CF-Access-Client-Id') || '';
    const clientSecret = request.headers.get('CF-Access-Client-Secret') || '';
    const auth = request.headers.get('Authorization') || '';

    if (
      clientId !== env.CF_CLIENT_ID ||
      clientSecret !== env.CF_CLIENT_SECRET ||
      auth !== `Bearer ${env.CB16_WORKER_TOKEN}`
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

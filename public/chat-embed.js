(function() {
  function initChatEmbed() {
    // Configuração: categoria inicial
    var initialCategory = window.ChatRagCategory || 'ecampus';
    var chatUrl = 'http://localhost:3001/chat-embed?category=' + encodeURIComponent(initialCategory);

    // Cria o botão flutuante
    var btn = document.createElement('button');
    btn.id = 'chat-rag-float-btn';
    btn.style.position = 'fixed';
    btn.style.bottom = '32px';
    btn.style.right = '32px';
    btn.style.zIndex = '99999';
    btn.style.background = '#2563eb';
    btn.style.color = '#fff';
    btn.style.border = 'none';
    btn.style.borderRadius = '50%';
    btn.style.width = '64px';
    btn.style.height = '64px';
    btn.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    btn.style.cursor = 'pointer';
    btn.style.fontSize = '32px';
    btn.innerHTML = '💬';

    // Cria o iframe do chat (inicialmente oculto)
    var iframe = document.createElement('iframe');
    iframe.id = 'chat-rag-iframe';
    iframe.src = chatUrl;
    iframe.style.position = 'fixed';
    iframe.style.bottom = '110px';
    iframe.style.right = '32px';
    iframe.style.width = '370px';
    iframe.style.height = '540px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '16px';
    iframe.style.boxShadow = '0 4px 24px rgba(0,0,0,0.25)';
    iframe.style.zIndex = '99999';
    iframe.style.display = 'none';
    iframe.style.background = '#fff';

    // Toggle do chat
    btn.onclick = function() {
      iframe.style.display = (iframe.style.display === 'none') ? 'block' : 'none';
    };

    // Fecha o chat ao clicar fora do iframe
    document.addEventListener('click', function(e) {
      if (iframe.style.display === 'block' && !iframe.contains(e.target) && e.target !== btn) {
        iframe.style.display = 'none';
      }
    });

    document.body.appendChild(btn);
    document.body.appendChild(iframe);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChatEmbed);
  } else {
    initChatEmbed();
  }
})();
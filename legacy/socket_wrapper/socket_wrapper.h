#ifndef SOCKET_WRAPPER
#define SOCKET_WRAPPER

#include <string>

int wrapped_connect(int sd, const char * address);
int wrapped_bind(int sd, const char * address);
int wrapped_accept(int sd);
int wrapped_send(int sd, char * buffer);
std::string wrapped_recv(int sd, int buffer_length);

#endif

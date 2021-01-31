#include <string>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int wrapped_connect(int sd, const char * address) {
    struct sockaddr_un serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sun_family = AF_UNIX;
    strcpy(serveraddr.sun_path, address);
    
    return connect(sd, (struct sockaddr *)&serveraddr, SUN_LEN(&serveraddr));
}

int wrapped_bind(int sd, const char * address) {
    struct sockaddr_un serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sun_family = AF_UNIX;
    strcpy(serveraddr.sun_path, address);

    return bind(sd, (struct sockaddr *)&serveraddr, SUN_LEN(&serveraddr));
}

int wrapped_accept(int sd) {
    return accept(sd, NULL, NULL);
}

int wrapped_send(int sd, char * buffer) {
    return send(sd, buffer, sizeof(buffer), 0);
}

const char * wrapped_recv(int sd, int buffer_length) {
    char buffer[buffer_length];
    recv(sd, buffer, sizeof(buffer), 0);
    std::string s(buffer);
    const char * r = s.c_str();
    return r;
}
